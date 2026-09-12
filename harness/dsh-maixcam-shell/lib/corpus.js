/**
 * 语料与索引的加载、校验与检索。
 *
 * ## 边界：重活在构建期，轻活在运行期
 *
 * 抓语料、切分、抽 API 符号、建索引都是**构建期**的一次性工作，由教学主线那套
 * Python 管线产出三个与检索实现无关的产物：
 *
 * ```
 * corpus/processed/chunks.jsonl        切分结果（含 heading_path 与 meta）
 * corpus/processed/api_symbols.jsonl   API 符号与精确签名（防幻觉白名单）
 * indexes/vector/vectors.npy + vectors_meta.json   稠密向量
 * indexes/index_meta.json              语料指纹与统计
 * ```
 *
 * 运行期只做**查询**：把问题嵌成向量、算余弦、取前 k 条。没有索引构建、没有分词器、
 * 没有第二个进程。
 *
 * ## 为什么这里只有稠密检索
 *
 * 稀疏那一路（BM25）在 Node 侧需要一个**与 Python 完全一致的分词器**才能复用
 * `indexes/bm25.json` —— 而那个分词器是 `zh_identifier`（jieba 分中文词）。
 * 在 JS 里复刻它就得背一份词典或引一个原生扩展，两者都不划算。
 *
 * 所以这一档的答案是：**稀疏留在 Python，运行期走嵌入**。嵌入模型（bge-m3，
 * 本地 Ollama）本身就是一个成熟组件，这里只写调用与余弦 —— 这正是「不重造检索」
 * 的意思。混合检索与重排留在后面（见 docs/design/06 §3.3）。
 *
 * ## 校验比检索重要
 *
 * 语料和索引是**两份可能各自更新的产物**。它们一旦对不上，检索会安静地返回
 * 牛头不对马嘴的片段 —— 没有报错，只有错答案。所以这里在加载时逐条核对
 * `vectors_meta.chunk_ids` 与 `chunks.jsonl` 的 `chunk_id`，不一致就**拒绝启动**
 * 而不是降级。宁可明确说不干，也不要安静地答错。
 *
 * @module dsh-maixcam-shell/corpus
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

/** 语料产物缺失、损坏，或彼此对不上时抛这个。 */
export class CorpusError extends Error {
	/**
	 * @param message - 给人看的原因，要能直接拿去修。
	 */
	constructor(message) {
		super(message)
		this.name = 'CorpusError'
	}
}

/**
 * 解析 `.npy` 文件头。
 *
 * `.npy` v1.0 的布局是：6 字节魔数 + 2 字节版本 + 2 字节小端头长度 + 头 + 数据。
 * 头本身是一段 Python 字典字面量，这里只需认出 `descr` 与 `shape` 两个键。
 *
 * @param buf - 整个文件的字节。
 * @param label - 出错时用来指明是哪个文件。
 * @returns `{ data, rows, cols }`，`data` 是行优先的 float32。
 */
function parseNpy(buf, label) {
	if (buf.length < 10 || buf.toString('latin1', 0, 6) !== '\x93NUMPY') {
		throw new CorpusError(`${label} 不是 .npy 文件（魔数不对）`)
	}
	const headerLen = buf.readUInt16LE(8)
	const header = buf.toString('latin1', 10, 10 + headerLen)
	const offset = 10 + headerLen

	const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1]
	if (descr !== '<f4') {
		throw new CorpusError(`${label} 的 dtype 是 ${descr ?? '未知'}，只支持 '<f4'`)
	}
	const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1]
	const dims = (shape ?? '').split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n))
	if (dims.length !== 2) {
		throw new CorpusError(`${label} 的 shape 不是二维：(${shape ?? ''})`)
	}
	const [rows, cols] = dims
	const need = rows * cols * 4
	if (buf.length - offset < need) {
		throw new CorpusError(`${label} 数据不足：需要 ${need} 字节，只有 ${buf.length - offset}`)
	}

	// 复制一份再建视图：Buffer 的内存池不保证 4 字节对齐，而 TypedArray 要求对齐。
	const slice = buf.buffer.slice(buf.byteOffset + offset, buf.byteOffset + offset + need)
	return { data: new Float32Array(slice), rows, cols }
}

/**
 * 逐行读一个 JSONL 文件。
 *
 * @param path - 文件路径。
 * @param label - 出错时用来指明是哪个文件。
 * @returns 解析出的对象数组。
 */
function readJsonl(path, label) {
	let text
	try {
		text = readFileSync(path, 'utf8')
	} catch (err) {
		throw new CorpusError(`读不到 ${label}：${path}（${err.message}）`)
	}
	const out = []
	const lines = text.split('\n')
	for (let i = 0; i < lines.length; i += 1) {
		const line = lines[i].trim()
		if (line === '') continue
		try {
			out.push(JSON.parse(line))
		} catch (err) {
			throw new CorpusError(`${label} 第 ${i + 1} 行不是合法 JSON：${err.message}`)
		}
	}
	return out
}

/**
 * 读一个 JSON 文件。
 *
 * @param path - 文件路径。
 * @param label - 出错时用来指明是哪个文件。
 * @returns 解析出的值。
 */
function readJson(path, label) {
	try {
		return JSON.parse(readFileSync(path, 'utf8'))
	} catch (err) {
		throw new CorpusError(`读不到或解析不了 ${label}：${path}（${err.message}）`)
	}
}

/**
 * 加载并校验一份语料。
 *
 * @param options - 加载选项。
 * @param options.root - 仓库根目录（语料产物都在它下面）。
 * @returns 语料句柄，含 `search` 之外的只读数据与 `stats`。
 */
export function loadCorpus({ root }) {
	const metaPath = join(root, 'indexes', 'index_meta.json')
	const meta = readJson(metaPath, 'index_meta.json')

	const chunks = readJsonl(join(root, 'corpus', 'processed', 'chunks.jsonl'), 'chunks.jsonl')
	if (chunks.length === 0) throw new CorpusError('chunks.jsonl 是空的')

	if (typeof meta.chunks === 'number' && meta.chunks !== chunks.length) {
		throw new CorpusError(
			`语料与索引对不上：index_meta.json 说 ${meta.chunks} 条切片，`
				+ `chunks.jsonl 里有 ${chunks.length} 条。重新构建索引（python -m maixrag corpus index）。`,
		)
	}

	const vectorsMeta = readJson(
		join(root, 'indexes', 'vector', 'vectors_meta.json'),
		'vectors_meta.json',
	)
	const ids = vectorsMeta.chunk_ids
	if (!Array.isArray(ids)) throw new CorpusError('vectors_meta.json 里没有 chunk_ids')

	const { data, rows, cols } = parseNpy(
		readFileSync(join(root, 'indexes', 'vector', 'vectors.npy')),
		'vectors.npy',
	)
	if (rows !== chunks.length || ids.length !== chunks.length) {
		throw new CorpusError(
			`语料与向量对不上：chunks ${chunks.length} 条，vectors ${rows} 条，chunk_ids ${ids.length} 条。`,
		)
	}
	if (typeof vectorsMeta.dim === 'number' && vectorsMeta.dim !== cols) {
		throw new CorpusError(
			`向量维度对不上：vectors_meta.json 说 ${vectorsMeta.dim}，vectors.npy 是 ${cols}。`,
		)
	}

	// 逐条核对顺序。这是最有价值的一条检查：向量是按 chunk_id 顺序存的，
	// 一旦两个产物更新的先后不一致，索引就会整体错位 —— 而且不会报任何错。
	for (let i = 0; i < chunks.length; i += 1) {
		if (ids[i] !== chunks[i].chunk_id) {
			throw new CorpusError(
				`语料与向量错位：第 ${i} 条，chunks 是 ${chunks[i].chunk_id}，`
					+ `vectors 是 ${ids[i]}。重新构建索引。`,
			)
		}
	}

	// 预先算好每一行的模长，检索时就不用重复开方。
	const norms = new Float32Array(rows)
	for (let r = 0; r < rows; r += 1) {
		let sum = 0
		const base = r * cols
		for (let c = 0; c < cols; c += 1) {
			const v = data[base + c]
			sum += v * v
		}
		norms[r] = Math.sqrt(sum)
	}

	const symbols = readJsonl(
		join(root, 'corpus', 'processed', 'api_symbols.jsonl'),
		'api_symbols.jsonl',
	)
	const byQualname = new Map()
	const byName = new Map()
	for (const sym of symbols) {
		if (sym.qualname) byQualname.set(sym.qualname, sym)
		if (sym.name) {
			const list = byName.get(sym.name)
			if (list === undefined) byName.set(sym.name, [sym])
			else list.push(sym)
		}
	}

	let roster = []
	try {
		roster = readFileSync(join(root, 'corpus', 'processed', 'api_roster.txt'), 'utf8')
			.split('\n')
			.map((s) => s.trim())
			.filter((s) => s !== '')
	} catch {
		// 名单是可选的：没有它就不做白名单校验，而不是整个检索起不来。
		roster = []
	}

	const kinds = {}
	for (const c of chunks) kinds[c.kind ?? 'unknown'] = (kinds[c.kind ?? 'unknown'] ?? 0) + 1

	return {
		root,
		meta,
		chunks,
		vectors: data,
		dim: cols,
		norms,
		symbols,
		byQualname,
		byName,
		roster,
		stats: {
			chunks: chunks.length,
			dim: cols,
			kinds,
			symbols: symbols.length,
			roster: roster.length,
			embeddingModel: meta.embedding_model ?? vectorsMeta.model_name ?? '未知',
			corpusFingerprint: meta.corpus_fingerprint ?? '',
		},
	}
}

/**
 * 把一段文本嵌成向量（OpenAI 兼容的 `/embeddings`）。
 *
 * 这里刻意只用最朴素的 HTTP 调用：嵌入服务是**外部依赖**，它挂掉是正常运行中
 * 会发生的事，所以失败要带着原因抛出来，由上层决定降级还是报错。
 *
 * @param options - 端点与超时。
 * @param options.baseUrl - 例如 `http://127.0.0.1:11434/v1`。
 * @param options.model - 例如 `bge-m3`。
 * @param options.apiKey - 可选的 Bearer。
 * @param options.timeoutMs - 超时毫秒数。
 * @param text - 要嵌入的文本。
 * @returns 归一化之前的原始向量。
 */
export async function embed(options, text) {
	const url = `${options.baseUrl.replace(/\/+$/, '')}/embeddings`
	const controller = new AbortController()
	const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 20000)
	let res
	try {
		res = await fetch(url, {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
			},
			body: JSON.stringify({ model: options.model, input: [text] }),
			signal: controller.signal,
		})
	} catch (err) {
		throw new CorpusError(`嵌入服务连不上：${url}（${err.message}）`)
	} finally {
		clearTimeout(timer)
	}
	if (!res.ok) {
		throw new CorpusError(`嵌入服务返回 ${res.status}：${(await res.text()).slice(0, 200)}`)
	}
	const body = await res.json()
	const vec = body?.data?.[0]?.embedding
	if (!Array.isArray(vec)) throw new CorpusError('嵌入服务没有返回 data[0].embedding')
	return vec
}

/**
 * 用余弦相似度取前 k 条切片。
 *
 * @param corpus - {@link loadCorpus} 的返回值。
 * @param queryVec - 查询向量（长度必须等于 `corpus.dim`）。
 * @param k - 取几条。
 * @returns `[{ index, score }]`，按分数从高到低。
 */
export function topK(corpus, queryVec, k) {
	if (queryVec.length !== corpus.dim) {
		throw new CorpusError(`查询向量维度是 ${queryVec.length}，索引是 ${corpus.dim}`)
	}
	let qnorm = 0
	for (let i = 0; i < queryVec.length; i += 1) qnorm += queryVec[i] * queryVec[i]
	qnorm = Math.sqrt(qnorm)
	if (qnorm === 0) throw new CorpusError('查询向量是零向量')

	const { vectors, norms, dim, chunks } = corpus
	const scored = new Array(chunks.length)
	for (let r = 0; r < chunks.length; r += 1) {
		const base = r * dim
		let dot = 0
		for (let c = 0; c < dim; c += 1) dot += vectors[base + c] * queryVec[c]
		scored[r] = { index: r, score: norms[r] === 0 ? 0 : dot / (norms[r] * qnorm) }
	}
	scored.sort((a, b) => b.score - a.score)
	return scored.slice(0, k)
}

/**
 * 把一条切片压成前端和工具都能直接用的形状。
 *
 * 只取叶子字段，绝不把内部对象整个丢出去 —— 语料里带着 `meta` 与整段正文，
 * 全量外传既浪费又容易漏出不该出现的东西。
 *
 * @param corpus - 语料句柄。
 * @param hit - `{ index, score }`。
 * @returns 一条证据。
 */
export function toEvidence(corpus, hit) {
	const chunk = corpus.chunks[hit.index]
	const meta = chunk.meta ?? {}
	return {
		chunk_id: chunk.chunk_id,
		doc_id: chunk.doc_id,
		doc_title: meta.doc_title ?? '',
		kind: chunk.kind ?? 'prose',
		heading_path: chunk.heading_path ?? [],
		score: Number(hit.score.toFixed(4)),
		url: meta.url ?? '',
		module: meta.module ?? '',
		text: chunk.text ?? '',
	}
}

/**
 * 查一个 API 符号的精确签名。
 *
 * @param corpus - 语料句柄。
 * @param name - 限定名（`camera.AeMode`）或裸名（`AeMode`）。
 * @returns `{ found, exact, candidates, inRoster }`。
 */
export function lookupSymbol(corpus, name) {
	const key = String(name ?? '').trim()
	if (key === '') return { found: false, exact: null, candidates: [], inRoster: false }

	const exact = corpus.byQualname.get(key) ?? null
	if (exact !== null) {
		return {
			found: true,
			exact,
			candidates: [],
			inRoster: corpus.roster.includes(key) || corpus.roster.includes(exact.name),
		}
	}

	const candidates = corpus.byName.get(key) ?? []
	return {
		found: candidates.length > 0,
		exact: null,
		candidates,
		inRoster: corpus.roster.includes(key),
	}
}
