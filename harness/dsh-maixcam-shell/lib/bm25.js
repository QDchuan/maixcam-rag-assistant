/**
 * BM25 稀疏检索：在 Node 侧从 `chunks.jsonl` 现建索引。
 *
 * ## 为什么不是「复用 indexes/bm25.json」
 *
 * 那份产物是**教学主线那个 Python 实现的分词器**（`zh_identifier`，jieba 分中文词）
 * 算出来的统计量。要在 Node 里用它，就得先复刻那个分词器 —— 这件事我原先写成了
 * 「所以运行期只能走稠密」，那是错的。
 *
 * 正确的看法是：`indexes/bm25.json` 是**某个实现的派生物**，
 * 而 `chunks.jsonl` 才是**与实现无关的中间产物**。换实现就该从后者重建，
 * 而不是去迁就前者的分词约定 —— 当初把两者分开，要的就是今天能这样换。
 *
 * ## 分词：ASCII 整词 + 中文二元组
 *
 * 不做分词器，做两件事：
 *
 * 1. **ASCII 词/标识符整段保留** —— `find_blobs`、`maix.image`、`pwm` 全是整词。
 *    这一条对本项目最关键：中英混合标识符是最容易翻车的地方，用整词才不会把
 *    `find_blobs` 切成 `find` + `blobs` 而丢掉精确匹配。
 * 2. **中文切二元组**（bigram）—— 「舵机」→ `舵机`，「寻找色块」→ `寻找`,`找色`,`色块`。
 *    没有词典也能拿到接近分词的召回，代价是索引大一些（这里完全可以接受）。
 *
 * 这是个**刻意的取舍**：不追求与 Python 侧分词一致，只追求召回够用。
 * 两边不一致没关系 —— 它们是两套独立的实现，没有谁必须迁就谁。
 *
 * @module dsh-maixcam-shell/bm25
 */

/** BM25 词频饱和参数。 */
const K1 = 1.5
/** BM25 文档长度归一化参数。 */
const B = 0.75

/** 一个 ASCII 词或点分标识符（`find_blobs`、`maix.image.Image`、`pwm`）。 */
const ASCII_RUN = /[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*/g
/** 一段连续中文（含常见 CJK 区）。 */
const CJK_RUN = /[\u3400-\u4dbf\u4e00-\u9fff]+/g

/**
 * 把一段文本切成检索用的词项。
 *
 * @param text - 原始文本。
 * @returns 词项数组（含重复，供词频统计）。
 */
export function tokenize(text) {
	const s = String(text ?? '')
	const out = []

	for (const m of s.matchAll(ASCII_RUN)) out.push(m[0].toLowerCase())

	CJK_RUN.lastIndex = 0
	for (const m of s.matchAll(CJK_RUN)) {
		const run = m[0]
		if (run.length === 1) {
			out.push(run)
			continue
		}
		for (let i = 0; i + 1 < run.length; i += 1) out.push(run.slice(i, i + 2))
	}

	return out
}

/**
 * 从切片数组建一份 BM25 索引。
 *
 * 索引结构刻意保持朴素：`tf` 是按切片存的词频表，`df` 是词项 → 出现过的切片数。
 * 3838 条切片、正文约 2.9 MB，建索引在百毫秒量级，不需要倒排链之外的东西。
 *
 * @param chunks - `chunks.jsonl` 解析出的切片数组。
 * @returns `{ search, stats }`。
 */
export function buildBm25(chunks) {
	const n = chunks.length
	const tf = new Array(n)
	const len = new Float64Array(n)
	const df = new Map()
	/** 词项 → 该词项出现过的切片下标（只在算分时需要，用于跳过无关切片）。 */
	const postings = new Map()

	for (let i = 0; i < n; i += 1) {
		const chunk = chunks[i]
		// 标题路径是强信号：同一篇文档下的切片靠它区分；doc_title 让「找整篇」也能命中。
		const head = (chunk.heading_path ?? []).join(' ')
		const title = chunk.meta?.doc_title ?? ''
		const tokens = tokenize(`${title} ${head} ${chunk.text ?? ''}`)
		const counts = new Map()
		for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1)

		tf[i] = counts
		len[i] = tokens.length
		for (const t of counts.keys()) {
			df.set(t, (df.get(t) ?? 0) + 1)
			const list = postings.get(t)
			if (list === undefined) postings.set(t, [i])
			else list.push(i)
		}
	}

	let avgLen = 0
	for (let i = 0; i < n; i += 1) avgLen += len[i]
	avgLen = n === 0 ? 0 : avgLen / n

	/**
	 * 跑一次稀疏检索。
	 *
	 * 只对**至少命中一个词项**的切片打分：靠倒排表取候选，不做全量扫描。
	 * 这一条在 3838 条上差别不大，但它是让稀疏能跟着语料长大的前提。
	 *
	 * @param query - 查询文本。
	 * @param k - 取几条。
	 * @returns `[{ index, score }]`，按分数降序。
	 */
	function search(query, k) {
		const qTokens = tokenize(query)
		if (qTokens.length === 0) return []

		const scores = new Map()
		for (const t of new Set(qTokens)) {
			const list = postings.get(t)
			if (list === undefined) continue
			const dfi = df.get(t) ?? 0
			// BM25 的 IDF（加 1 平滑，避免高频词给出负分）。
			const idf = Math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))
			for (const i of list) {
				const f = tf[i].get(t) ?? 0
				const norm = 1 - B + (B * len[i]) / (avgLen || 1)
				const add = idf * ((f * (K1 + 1)) / (f + K1 * norm))
				scores.set(i, (scores.get(i) ?? 0) + add)
			}
		}

		return [...scores.entries()]
			.map(([index, score]) => ({ index, score }))
			.sort((a, b) => b.score - a.score)
			.slice(0, k)
	}

	return {
		search,
		stats: { documents: n, terms: df.size, avgLength: Number(avgLen.toFixed(1)) },
	}
}

/**
 * Reciprocal Rank Fusion：把多路结果按**名次**融合。
 *
 * 用名次而不是分数，是因为稠密（余弦，0~1）与稀疏（BM25，无上界）的分数**不可比**——
 * 直接加权求和就得先做归一化，而归一化系数在不同查询上并不稳定。
 * RRF 绕开了这件事：只看「排第几」。
 *
 * @param lists - 多路 `[{ index, score }]`，各自已按分数降序。
 * @param k - 融合后取几条。
 * @param rankConstant - RRF 常数，通常 60。
 * @returns `[{ index, score, from }]`，`from` 记录每一路各自给了第几名。
 */
export function rrf(lists, k, rankConstant = 60) {
	const fused = new Map()
	const names = ['dense', 'sparse']

	lists.forEach((list, li) => {
		list.forEach((hit, rank) => {
			const cur = fused.get(hit.index) ?? { index: hit.index, score: 0, from: {} }
			cur.score += 1 / (rankConstant + rank + 1)
			cur.from[names[li] ?? `list${li}`] = rank + 1
			fused.set(hit.index, cur)
		})
	})

	return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, k)
}
