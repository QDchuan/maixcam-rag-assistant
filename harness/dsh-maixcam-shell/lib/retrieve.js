/**
 * 检索层 —— **由 LangChain.js 驱动**，不再自研。
 *
 * ## 为什么换掉自研那一层
 *
 * 这一层原先是我手写的：自己的稠密余弦、自己的 BM25、自己的分词器、自己的 RRF、
 * 自己的查询扩展提示词、自己拍的阈值。`docs/design/06` §3.3.1 早就写明
 * 「RAG 不自研，从开源方案里挑一个」，我一直没照做，理由是「就几十行」。
 * 结果是每一轮都靠肉眼看一两个问题调参，没有一项是经过大量场景验证的。
 *
 * 现在换成现成的，逐项对应：
 *
 * | 原来我手写的 | 现在用的 |
 * | --- | --- |
 * | `embed()` + 余弦 top-k | `OpenAIEmbeddings`（指向本机 Ollama）+ `MemoryVectorStore` |
 * | `bm25.js`（自写分词器 + BM25） | `BM25Retriever`（`@langchain/community`，自带分词） |
 * | `rrf()` | `EnsembleRetriever`（分数归一化后加权融合） |
 * | `expand.js` + 逐面循环 | `MultiQueryRetriever`（LLM 生成多个查询、各自检索、去重合并） |
 *
 * ## 唯一自己接的一处：向量库
 *
 * `MemoryVectorStore.addVectors()` 直接吃**已经算好的**向量 —— 教学主线那套 Python 管线
 * 产出的 `vectors.npy` 原样灌进去，一个字都不用重新嵌入。
 * 这不是自研，是框架设计好的插槽（`VectorStore` 接口）。
 *
 * ## 仍然保留的
 *
 * 语料产物（`chunks.jsonl` / `api_symbols.jsonl`）仍是与实现无关的中间产物 ——
 * 这正是当初把「语料」和「检索实现」分开的价值：换框架不用动语料。
 *
 * @module dsh-maixcam-shell/retrieve
 */

import { readFileSync } from 'node:fs'
import { BM25Retriever } from '@langchain/community/retrievers/bm25'
import { Document } from '@langchain/core/documents'
import { MultiQueryRetriever } from '@langchain/classic/retrievers/multi_query'
import { EnsembleRetriever } from '@langchain/classic/retrievers/ensemble'
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'

/** 一份语料 + 由现成组件拼成的检索链。 */
export class MaixcamRetriever {
	/**
	 * @param options - 语料与端点。
	 * @param options.chunks - `chunks.jsonl` 解析出的切片。
	 * @param options.vectorsPath - `vectors.npy` 路径。
	 * @param options.vectorsMetaPath - `vectors_meta.json` 路径（提供 chunk_ids 顺序）。
	 * @param options.embed - 嵌入端点（OpenAI 兼容，指向本机 Ollama）。
	 * @param options.llm - 对话端点（OpenAI 兼容，指向 DeepSeek）；缺省则不做多查询扩展。
	 * @param options.docsPerQuery - 每个查询取几篇。默认 8。
	 * @param options.queryCount - 多查询扩展生成几个查询。默认 5。
	 */
	constructor(options) {
		this.options = { docsPerQuery: 8, queryCount: 5, ...options }
		this.documents = this.options.chunks.map(
			(chunk) =>
				new Document({
					pageContent: chunk.text ?? '',
					metadata: {
						chunk_id: chunk.chunk_id,
						doc_id: chunk.doc_id,
						doc_title: chunk.meta?.doc_title ?? chunk.doc_id,
						module: chunk.meta?.module ?? '',
						url: chunk.meta?.url ?? '',
						heading_path: (chunk.heading_path ?? []).join(' › '),
					},
				}),
		)
	}

	/**
	 * 惰性建链。第一次用到才读向量、建索引 —— 宿主启动不该被 15 MB 的向量拖住。
	 *
	 * @returns 建好的检索链。
	 */
	async chain() {
		if (this.chainPromise !== undefined) return this.chainPromise
		this.chainPromise = (async () => {
			const { embed, llm, docsPerQuery } = this.options

			// ── 稠密：把我们**已经算好的**向量灌进内存向量库 ──────────────────
			const embeddings = new OpenAIEmbeddings({
				model: embed.model,
				apiKey: embed.apiKey || 'not-needed',
				configuration: { baseURL: embed.baseUrl },
			})
			const store = new MemoryVectorStore(embeddings)
			const vectors = readVectors(this.options.vectorsPath, this.options.vectorsMetaPath)
			if (vectors.length !== this.documents.length) {
				throw new Error(
					`语料与向量对不上：切片 ${this.documents.length} 条，向量 ${vectors.length} 条 —— 拒绝用错位的数据检索。`,
				)
			}
			await store.addVectors(vectors, this.documents)

			// ── 稀疏：现成 BM25，自带分词（不用我再复刻一遍） ──────────────────
			const bm25 = await BM25Retriever.fromDocuments(this.documents, { k: docsPerQuery })

			// ── 融合：现成的加权融合，替掉我那个 RRF ──────────────────────────
			const ensemble = new EnsembleRetriever({
				retrievers: [store.asRetriever(docsPerQuery), bm25],
				weights: [0.5, 0.5],
			})

			// ── 查询扩展：现成的 MultiQueryRetriever，替掉我那个 expand.js ─────
			const retriever =
				llm === undefined
					? ensemble
					: MultiQueryRetriever.fromLLM({
							llm: new ChatOpenAI({
								model: llm.model,
								apiKey: llm.apiKey,
								configuration: { baseURL: llm.baseUrl },
								temperature: 0,
							}),
							retriever: ensemble,
							queryCount: this.options.queryCount,
							// 提示词交给框架自带的那个 —— 它就是为这件事调的。
						})
			return { retriever, ensemble, bm25, store }
		})()
		return this.chainPromise
	}

	/**
	 * 检索一次。
	 *
	 * @param query - 用户的问题。
	 * @returns `{ docs, usedMultiQuery }`；`docs` 已按 `doc_id` 去重（一篇只留最相关的一片）。
	 */
	async search(query) {
		const { retriever } = await this.chain()
		const hits = await retriever.invoke(query)
		const seen = new Set()
		const docs = []
		for (const doc of hits) {
			const id = doc.metadata.doc_id
			if (seen.has(id)) continue
			seen.add(id)
			docs.push({
				chunk_id: doc.metadata.chunk_id,
				doc_id: id,
				doc_title: doc.metadata.doc_title,
				module: doc.metadata.module,
				url: doc.metadata.url,
				heading_path: doc.metadata.heading_path.split(' › ').filter(Boolean),
				text: doc.pageContent,
			})
		}
		return { docs, usedMultiQuery: this.options.llm !== undefined }
	}
}

/**
 * 读 `.npy` 并转成 LangChain 要的 `number[][]`。
 *
 * `.npy` v1.0 布局：6 字节魔数 + 2 字节版本 + 2 字节小端头长度 + 头 + 数据。
 *
 * @param path - `vectors.npy` 路径。
 * @param metaPath - `vectors_meta.json` 路径。
 * @returns 每行一个向量。
 */
function readVectors(path, metaPath) {
	const meta = JSON.parse(readFileSync(metaPath, 'utf8'))
	const buf = readFileSync(path)
	const headerLen = buf.readUInt16LE(8)
	const header = buf.toString('latin1', 10, 10 + headerLen)
	const offset = 10 + headerLen
	const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1] ?? ''
	const [rows, cols] = shape.split(',').map((s) => Number(s.trim())).filter(Number.isFinite)
	const out = []
	for (let r = 0; r < rows; r += 1) {
		const row = new Array(cols)
		for (let c = 0; c < cols; c += 1) row[c] = buf.readFloatLE(offset + (r * cols + c) * 4)
		out.push(row)
	}
	if (Array.isArray(meta.chunk_ids) && meta.chunk_ids.length !== rows) {
		throw new Error('vectors_meta.json 的 chunk_ids 数与 vectors.npy 行数不一致')
	}
	return out
}
