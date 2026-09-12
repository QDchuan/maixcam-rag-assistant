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
	toEvidence,
	topK,
} from './corpus.js'
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
	 * 跑一次检索。路由和 agent 工具共用这一条路径 —— 只有一份实现，
	 * 界面看到的和模型看到的就一定是同一件事。
	 *
	 * @param query - 查询词。
	 * @param k - 取几条。
	 * @returns 与 `/search` 相同形状的结果（失败时抛错）。
	 */
	async function search(query, k) {
		const c = await ensureCorpus()
		const started = Date.now()
		const vector = await embed(
			{
				baseUrl: settings.embedBaseUrl,
				model: settings.embedModel,
				apiKey: settings.embedApiKey,
				timeoutMs: settings.embedTimeoutMs,
			},
			query,
		)
		const embedMs = Date.now() - started
		return {
			query,
			k,
			strategy: 'dense',
			embedMs,
			totalMs: Date.now() - started,
			hits: topK(c, vector, k).map((hit) => toEvidence(c, hit)),
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

		try {
			return json({ ok: true, ...(await search(query, k)) })
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
			search: async (query, k) => {
				try {
					return await search(query, k)
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
