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

		let c
		try {
			c = await ensureCorpus()
		} catch (error) {
			return json({ ok: false, error: reason(error) }, 503)
		}

		const started = Date.now()
		try {
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
			const hits = topK(c, vector, k).map((hit) => toEvidence(c, hit))
			return json({
				ok: true,
				query,
				k,
				strategy: 'dense',
				embedMs,
				totalMs: Date.now() - started,
				hits,
			})
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

	ctx.logger?.info?.(`dsh-maixcam-shell: 检索路由就绪（${PREFIX}）`)
}
