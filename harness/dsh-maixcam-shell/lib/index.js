/**
 * `dsh-maixcam-shell` — 宿主半边。
 *
 * 这个包交付两件事：
 *
 * 1. **界面**（浏览器半边，`./client.js`）：把通用外壳换成 MaixCAM 助手的界面；
 * 2. **领域检索的接入点**（本文件）：读构建期产物，把「问一句 → 拿回带出处的证据」
 *    做成宿主路由，供界面直接调用。
 *
 * ## 为什么检索放在宿主
 *
 * 浏览器读不到仓库里的语料文件，也不该拿到嵌入服务的地址和密钥；而宿主已经在
 * 同一个进程里，读到就能用。更关键的是**降级要说得出原因**：嵌入服务没起来、
 * 语料和索引对不上，这些事实只有宿主知道，界面只负责把它显示出来。
 *
 * ## 一条硬约束：语料不匹配就拒绝服务，不降级
 *
 * 见 `corpus.js` 顶部。这里只体现结果：加载失败时 `/api/maixcam/status` 报出
 * 精确原因，`/api/maixcam/search` 返回 503 而不是空结果 —— 空结果会被误读成
 * 「知识库里没有」，那正是这个项目要治的病。
 *
 * @module dsh-maixcam-shell
 */

import {
	CorpusError,
	embed,
	loadCorpus,
	lookupSymbol,
	moduleCoverage,
	toEvidence,
	topK,
} from './corpus.js'
import { rrf } from './bm25.js'
import { expandQuery, readCredential } from './expand.js'
import { createCorpusTools, unavailable } from './tools.js'
import { checkSymbols } from './symbols.js'

/** 需要连接服务来注册 `/api` 路由。 */
export const inject = ['connection']

/** 路由前缀。界面与宿主靠它对齐。 */
const PREFIX = '/api/maixcam'

/** 配置默认值；补丁行里的 `config` 覆盖它们。 */
const DEFAULTS = {
	/** 仓库根目录：语料产物都在它下面。 */
	repoRoot: '',
	/** 嵌入服务（OpenAI 兼容）。默认本机 Ollama。 */
	embedBaseUrl: 'http://127.0.0.1:11434/v1',
	/** 嵌入模型，必须与建索引时用的一致。 */
	embedModel: 'bge-m3',
	/** 需要鉴权时的 Bearer；本机 Ollama 不需要。 */
	embedApiKey: '',
	/** 默认取几条证据。 */
	topK: 5,
	/** 单次嵌入调用的超时。 */
	embedTimeoutMs: 20000,
	/** 语义扩展用的 LLM（OpenAI 兼容）。留空则退回单查询。 */
	expandBaseUrl: '',
	/** 扩展用的模型名。 */
	expandModel: '',
	/** 从凭据文件里取的键名。 */
	expandApiKeyRef: 'DEEPSEEK_API_KEY',
	/** 凭据文件路径（本应用自己 home 下的那个）。 */
	credentialsFile: '',
	/** 扩展调用超时。 */
	expandTimeoutMs: 20000,

	/** 是否把检索能力注册成 agent 工具。 */
	enableTools: true,
	/** agent 工具的单次调用超时。 */
	toolTimeoutMs: 60000,
}

/** 不缓存：这些响应随语料与查询变化，缓存只会让人对不上号。 */
const NO_STORE = { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' }

/**
 * 造一个 JSON 响应。
 *
 * @param body - 要序列化的对象。
 * @param status - HTTP 状态码。
 * @returns 响应。
 */
function json(body, status = 200) {
	return new Response(JSON.stringify(body), { status, headers: NO_STORE })
}

/**
 * 把异常收敛成一条人话原因。
 *
 * @param error - 捕获到的异常。
 * @returns 原因字符串。
 */
function reason(error) {
	if (error instanceof CorpusError) return error.message
	return error?.message ?? String(error)
}

/**
 * 宿主插件体。
 *
 * @param ctx - 宿主插件上下文。
 * @param config - 补丁行里给的配置。
 */
export function apply(ctx, config) {
	const settings = { ...DEFAULTS, ...(config ?? {}) }
	ctx.logger?.info?.(
		`dsh-maixcam-shell: 外壳已挂载（语料根 ${settings.repoRoot === '' ? '未配置' : settings.repoRoot}）`,
	)

	/** 语料句柄；`undefined` 表示还没试过。 */
	let corpus
	/** 最近一次加载失败的原因；成功时清空。 */
	let loadError = null
	/** 正在进行的加载，用于把并发请求合成一次。 */
	let loading = null

	/**
	 * 惰性加载语料。第一次用到才读那 3 MB 正文和 15 MB 向量。
	 *
	 * 单飞（single-flight）：并发的两个请求共享同一次加载，失败则把结果丢掉，
	 * 让下一次请求可以重试 —— 用户修好索引之后不该还要重启进程。
	 *
	 * @returns 语料句柄。
	 */
	async function ensureCorpus() {
		if (corpus !== undefined) return corpus
		if (loading === null) {
			loading = (async () => {
				if (settings.repoRoot === '') {
					throw new CorpusError(
						'没有配置 repoRoot：补丁行里要给出仓库根目录，语料产物在它下面。'
							+ '重跑 python -m harness.install 可以自动写进去。',
					)
				}
				return loadCorpus({ root: settings.repoRoot })
			})()
		}
		try {
			corpus = await loading
			loadError = null
			ctx.logger?.info?.(
				`dsh-maixcam-shell: 语料已加载（${corpus.stats.chunks} 条切片，`
					+ `${corpus.stats.symbols} 个 API 符号，嵌入模型 ${corpus.stats.embeddingModel}）`,
			)
			return corpus
		} catch (error) {
			loading = null
			loadError = reason(error)
			ctx.logger?.warn?.(`dsh-maixcam-shell: 语料加载失败 —— ${loadError}`)
			throw error
		}
	}

	/**
	 * 注册一条路由，所有权交给当前 fiber。
	 *
	 * @param path - 完整路径。
	 * @param methods - HTTP 方法。
	 * @param requestBody - `'buffered'` 或 `'streaming'`。
	 * @param fetch - 处理函数。
	 */
	function route(path, methods, requestBody, fetch) {
		ctx.effect(
			() => ctx.connection.fetch.register({ path, methods, requestBody, fetch }),
			`dsh-maixcam-shell: ${path}`,
		)
	}

	/**
	 * 判断这是不是一道**设计类**问题（要搭一个系统，而不是问一个事实）。
	 *
	 * 判据刻意保守：同时出现「造东西」的动词和「一个东西」的名词才算。
	 * 「PWM 怎么用」不是设计题，「设计一个人脸跟随系统」是。
	 *
	 * @param query - 查询词。
	 * @returns 是否像设计类问题。
	 */
	function looksLikeDesignTask(query) {
		const q = String(query ?? '')
		return /设计|实现|做一个|做个|搭建|搭一个|开发|规划|方案/.test(q)
			&& /系统|方案|项目|装置|平台|机器人|云台|跟随|闭环/.test(q)
	}
	/**
	 * 跑一次**混合检索**：稠密 + 稀疏，RRF 融合。
	 *
	 * @param c - 语料句柄。
	 * @param text - 查询词。
	 * @param k - 取几条。
	 * @returns `{ hits, embedMs }`。
	 */
	async function hybrid(c, text, k) {
		const started = Date.now()
		const vector = await embed(
			{
				baseUrl: settings.embedBaseUrl,
				model: settings.embedModel,
				apiKey: settings.embedApiKey,
				timeoutMs: settings.embedTimeoutMs,
			},
			text,
		)
		const embedMs = Date.now() - started

		// 两路各多取一些再融合：只在 k 条上融合，会丢掉「另一路排第 8」的好结果。
		const pool = Math.max(k * 4, 24)
		const dense = topK(c, vector, pool)
		const sparse = c.bm25.search(text, pool)

		// 按 RRF 名次取候选，再**每篇文档最多留 2 条**。
		//
		// 实测过的问题：k=6 的 6 条里 5 条来自「人脸追踪2轴云台」同一篇 ——
		// 于是「6 条命中」其实只覆盖了 1 个来源，看着检索到了东西，其实没有广度。
		// 按文档限流之后，同样的 k 能覆盖更多篇。候选池同时放宽到 k×4（下限 24）。
		const PER_DOC_CAP = 2
		const perDoc = new Map()
		const hits = []
		for (const hit of rrf([dense, sparse], Math.max(k * 4, 24))) {
			const evidence = toEvidence(c, hit)
			const used = perDoc.get(evidence.doc_id) ?? 0
			if (used >= PER_DOC_CAP) continue
			perDoc.set(evidence.doc_id, used + 1)
			// 每一路各给了第几名 —— 让「这条为什么被选中」可见。
			hits.push({ ...evidence, from: hit.from, fused: Number(hit.score.toFixed(5)) })
			if (hits.length >= k) break
		}
		// 最高余弦：稠密那一路的绝对相似度，跨查询可比（RRF 分数不可比）。
		const topCosine = dense.length > 0 ? dense[0].score : 0
		return { hits, embedMs, topCosine }
	}

	/**
	 * 跑一次检索：**稠密 + 稀疏，RRF 融合**，并附一份模块覆盖度。
	 *
	 * 路由和 agent 工具共用这一条路径 —— 只有一份实现，界面看到的和模型看到的
	 * 就一定是同一件事。
	 *
	 * 为什么是两路：实测过稠密单路的病 —— 问「设计一个人脸跟随系统」，6 条命中
	 * **全部**落在 `projects`/`vision`，锚死在「人脸追踪2轴云台」一页；
	 * 而「舵机 PWM」「串口 uart」这些子系统查询与它的重合是 **0 条**。
	 * 稀疏那一路上靠的是**整词命中**（`pwm`、`uart`、`find_blobs`），
	 * 恰好补的就是稠密漏掉的那类关键词证据。
	 *
	 * ## `aspects`：把「拆子系统」变成可检查的
	 *
	 * 单条查询天然只覆盖一个面向，所以「怎么设计 X」必然锚死在那个像整题的页面上。
	 * 靠提示词求模型「记得想舵机」是软的；也试过一个自动推断「相关但未命中模块」的
	 * 办法，不成立 —— 模块名是英文、查询是中文，词面对不上，恒为空。
	 *
	 * 管用的做法是**让模型自己报子系统，然后逐个查、逐个报有没有证据**：
	 *
	 *     search_docs({ query: '设计一个人脸跟随系统',
	 *                   aspects: ['人脸检测', '舵机控制', '串口通信', '供电'] })
	 *
	 * 每个 aspect 都明写命中几条；**命中 0 条的那个就是信号** ——
	 * 「舵机控制：本地无证据」必须出现在工具结果里，也就必须出现在回答里。
	 * 与 `check_api_usage` 同一个手法：**让缺的东西显形，而不是不存在。**
	 *
	 * @param query - 查询词。
	 * @param k - 每个面向取几条。
	 * @param aspects - 可选的子系统清单；给了就逐个子系统检索。
	 * @returns 检索结果（失败时抛错）。
	 */
	async function search(query, k, aspects = []) {
		const c = await ensureCorpus()
		const started = Date.now()

		if (aspects.length === 0) {
			const { hits, embedMs } = await hybrid(c, query, k)

			// ── 前置语义扩展：**先让模型把问题拆开，再逐面做向量匹配** ──────────
			//
			// 为什么必须由模型来做：向量做不出这个关联（实测稠密检索把「人脸追踪2轴云台」
			// 端上来就停了）；而检索层自己用标题词面重合去猜也不成立（伪关联一堆）。
			// 模型天然会联想 —— 用户观察到它自己打出过「总线舵机 串口舵机 舵机供电 外部电源 电压」。
			//
			// 扩展失败一律退回单查询结果，不抛错：扩展是增强，不是必需。
			let facets = []
			if (settings.expandBaseUrl !== '' && settings.expandModel !== '') {
				const apiKey = readCredential(settings.credentialsFile, settings.expandApiKeyRef)
				const aspects = await expandQuery(
					{
						baseUrl: settings.expandBaseUrl,
						model: settings.expandModel,
						apiKey,
						timeoutMs: settings.expandTimeoutMs,
					},
					query,
				)
				if (aspects.length > 0) {
					for (const aspect of aspects) {
						const { hits: facetHits } = await hybrid(c, aspect, k)
						facets.push({ aspect, n: facetHits.length, coverage: moduleCoverage(c, facetHits), hits: facetHits })
					}
				}
			}

			return {
				query,
				k,
				strategy: 'hybrid(rrf: dense+bm25)',
				embedMs,
				totalMs: Date.now() - started,
				// 设计类问题却只查了一次 —— 交给渲染层把「你先拆子系统」这道门竖起来。
				needsAspects: facets.length === 0 && looksLikeDesignTask(query),
				expandedBy: facets.length > 0 ? settings.expandModel : '',
				facets,
				coverage: moduleCoverage(c, hits),
				hits,
			}
		}

		const facets = []
		for (const aspect of aspects) {
			const { hits, topCosine } = await hybrid(c, aspect, k)
			facets.push({
				aspect,
				n: hits.length,
				topCosine: Number(topCosine.toFixed(3)),
				// 置信度分三档，不做二值判断。阈值 0.60 是从实测分布里取的：
				// 真有证据的面落在 0.73~0.81，本地没有的面落在 0.52~0.56。
				confidence: topCosine >= 0.7 ? 'strong' : topCosine >= 0.6 ? 'medium' : 'weak',
				coverage: moduleCoverage(c, hits),
				hits,
			})
		}
		return {
			query,
			k,
			aspects,
			strategy: 'hybrid(rrf: dense+bm25) × 多面向',
			totalMs: Date.now() - started,
			// 检索永远会返回 k 条，所以「无证据」不能只看 n===0 —— 判据是置信度。
			empty: facets.filter((f) => f.confidence === 'weak').map((f) => f.aspect),
			facets,
		}
	}

	// ── 状态：界面靠它回答「你现在到底知道什么」 ─────────────────────────────
	route(`${PREFIX}/status`, ['GET'], 'buffered', async () => {
		const base = {
			ok: true,
			repoRoot: settings.repoRoot,
			embed: { baseUrl: settings.embedBaseUrl, model: settings.embedModel },
		}
		try {
			const c = await ensureCorpus()
			return json({ ...base, loaded: true, stats: c.stats })
		} catch (error) {
			// 状态路由本身不失败：它要能报出「为什么没就绪」，所以照样 200。
			return json({ ...base, loaded: false, error: reason(error), loadError })
		}
	})

	// ── 检索：问一句，拿回带出处的证据 ──────────────────────────────────────
	route(`${PREFIX}/search`, ['GET'], 'buffered', async (request) => {
		const url = new URL(request.url)
		const query = (url.searchParams.get('q') ?? '').trim()
		const k = Math.min(Math.max(Number(url.searchParams.get('k')) || settings.topK, 1), 20)
		if (query === '') return json({ ok: false, error: '缺少查询词 q' }, 400)
		const aspects = (url.searchParams.get('aspects') ?? '')
			.split(',')
			.map((s) => s.trim())
			.filter((s) => s !== '')

		try {
			return json({ ok: true, ...(await search(query, k, aspects)) })
		} catch (error) {
			// 检索失败要说清楚是嵌入服务的问题，而不是「没找到」。
			return json({ ok: false, error: reason(error), stage: 'embed' }, 503)
		}
	})

	// ── 符号：精确签名，防幻觉白名单的查询面 ────────────────────────────────
	route(`${PREFIX}/symbol`, ['GET'], 'buffered', async (request) => {
		const name = (new URL(request.url).searchParams.get('name') ?? '').trim()
		if (name === '') return json({ ok: false, error: '缺少符号名 name' }, 400)
		let c
		try {
			c = await ensureCorpus()
		} catch (error) {
			return json({ ok: false, error: reason(error) }, 503)
		}
		const result = lookupSymbol(c, name)
		return json({ ok: true, name, ...result })
	})

	// ── agent 工具：让助手真的去查，而不是凭记忆写 ──────────────────────────
	//
	// 这两个工具注册在**宿主平面**，因此对每一个会话都可见。对一个专用应用
	// （这个进程就是「MaixCAM 开发助手」）这正是想要的；若以后要与别的领域共用
	// 同一个进程，它们就该搬到 agent preset 里，并把检索服务放进 isolate realm ——
	// 那是 docs/design/06 §3.4 讲的平面归属问题。
	//
	// `tools` 服务是可选的：没有它时只少两个工具，路由照样工作。
	ctx.inject(['tools'], (toolsCtx) => {
		if (settings.enableTools !== true) {
			ctx.logger?.info?.('dsh-maixcam-shell: 检索工具已按配置关闭')
			return
		}
		const tools = createCorpusTools({
			search: async (query, k, aspects) => {
				try {
					return await search(query, k, aspects)
				} catch (error) {
					// 工具失败要说清原因，别让模型把「查不了」当成「没有」。
					throw new CorpusError(unavailable(error))
				}
			},
			lookup: async (name) => {
				try {
					return { name, ...lookupSymbol(await ensureCorpus(), name) }
				} catch (error) {
					throw new CorpusError(unavailable(error))
				}
			},
			check: async (code) => {
				try {
					const c = await ensureCorpus()
					// `assumeCode: true` —— 工具收到的是裸源码，没有 markdown 围栏。
					// 教学主线的事故 01 就出在这个参数上：不显式声明，校验器会对
					// 裸代码返回「检查了 0 个符号」，而 0 被当成通过。
					return checkSymbols(code, c.rosterSet, { assumeCode: true })
				} catch (error) {
					throw new CorpusError(unavailable(error))
				}
			},
			getTimeoutMs: () => settings.toolTimeoutMs,
		})
		for (const tool of tools) {
			const label = `dsh-maixcam-shell: tool ${tool.name ?? '?'}`
			ctx.effect(() => toolsCtx.tools.register(tool), label)
		}
		ctx.logger?.info?.(
			`dsh-maixcam-shell: 检索工具已注册（${tools.map((t) => t.name).join(' / ')}）`,
		)
	})

	ctx.logger?.info?.(`dsh-maixcam-shell: 检索路由就绪（${PREFIX}）`)
}
