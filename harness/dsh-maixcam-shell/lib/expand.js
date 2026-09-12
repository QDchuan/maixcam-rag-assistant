/**
 * 前置语义扩展：**先让 LLM 把问题拆开，再去做向量匹配**。
 *
 * ## 为什么必须由模型来做，而不是检索层自己猜
 *
 * 实测过的两件事：
 *
 * 1. 向量匹配做不出这个关联。问「设计一个二维云台人脸跟随系统」，稠密检索把
 *    「人脸追踪2轴云台」那一页端上来就停了 —— 它不会替你想到还要看舵机、PWM、引脚。
 * 2. 我自己用「文档标题词面重合」做过一版（`corpus.js` 里 `expandFacets` 的注释记着），
 *    **不成立**：中文切二元组之后「关键/检测」这类字面重合到处命中，吐出一堆伪关联。
 *
 * 而模型天然会做这个联想 —— 用户自己的观察最准：不限制它的时候，它上网搜的
 * 关键词极其精准（「总线舵机 串口舵机 舵机供电 外部电源 电压」就是它自己打出来的）。
 *
 * 所以扩展这一步**交给模型**，检索层只负责拿着扩展出来的词去查。
 * 这也符合本项目的分工：**机制自己写、关联交给模型、边界由工具报出来。**
 *
 * ## 它是一次很小的调用
 *
 * 提示词短、要求只输出一个 JSON 数组、`max_tokens` 也就两三百。
 * 一次问答多花几百 token，换来的是从「1 个来源」变成「5 个来源」。
 *
 * @module dsh-maixcam-shell/expand
 */

import { readFileSync } from 'node:fs'

/** 扩展器的提示词。要求苛刻是故意的：只要一个 JSON 数组，别的都不要。 */
const PROMPT = `你在为一个 MaixCAM / MaixPy 的开发助手准备检索词。

用户的问题会由你**先拆成若干必须分别查证的技术面向**，再拿这些词去检索本地文档库。
你的价值就在这里：向量检索只会返回「最像整题的那一页」，它不会想到这个系统还要用到
舵机、PWM、引脚、供电、通信这些东西 —— 而你会。

要求：
1. 只输出一个 JSON 数组，不要任何解释、不要代码围栏；
2. 每个元素是一个**短的检索词**（2~12 字），中文为主，专有名词保留英文；
3. 4~7 个面向，覆盖这个系统真正需要动手的各个部件，不要复述原问题；
4. 词要具体到能命中文档，例如「舵机 PWM 控制」「串口 UART 收发」「引脚分配 pinmap」「供电与电流」；
5. 如果问题本身很简单（只是在问一个 API 怎么用），就只返回 1~2 个词。

用户的问题：
`

/**
 * 从一个 OpenAI 兼容的 `/chat/completions` 响应里取出 JSON 数组。
 *
 * 模型偶尔会带上代码围栏或前后缀，所以先剥围栏、再截取第一个 `[` 到最后一个 `]`。
 *
 * @param content - 模型返回的正文。
 * @returns 解析出的字符串数组（可能为空）。
 */
export function parseFacets(content) {
	const text = String(content ?? '')
		.replace(/```[a-zA-Z]*\n?/g, '')
		.trim()
	const start = text.indexOf('[')
	const end = text.lastIndexOf(']')
	if (start < 0 || end <= start) return []
	let parsed
	try {
		parsed = JSON.parse(text.slice(start, end + 1))
	} catch {
		return []
	}
	if (!Array.isArray(parsed)) return []
	return parsed
		.map((x) => String(x ?? '').trim())
		.filter((x) => x !== '' && x.length <= 40)
		.slice(0, 8)
}

/**
 * 调一次 LLM，把问题扩展成一组检索面向。
 *
 * 失败一律**返回空数组**而不是抛错：扩展是增强，不是必需。
 * 检索层拿到空数组就走原来的单查询路径 —— 宁可退化，不能因为扩展器挂了就没有结果。
 *
 * @param options - 端点与超时。
 * @param options.baseUrl - OpenAI 兼容的基址，例如 `https://api.deepseek.com/v1`。
 * @param options.model - 模型名。
 * @param options.apiKey - Bearer。
 * @param options.timeoutMs - 超时。
 * @param query - 用户的问题。
 * @returns 面向数组；失败时为空数组。
 */
export async function expandQuery(options, query) {
	if (!options.apiKey || options.baseUrl === '') return []

	const controller = new AbortController()
	const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 20000)
	try {
		const res = await fetch(`${options.baseUrl.replace(/\/+$/, '')}/chat/completions`, {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				authorization: `Bearer ${options.apiKey}`,
			},
			body: JSON.stringify({
				model: options.model,
				temperature: 0,
				max_tokens: 300,
				messages: [{ role: 'user', content: PROMPT + query }],
			}),
			signal: controller.signal,
		})
		if (!res.ok) return []
		const body = await res.json()
		return parseFacets(body?.choices?.[0]?.message?.content)
	} catch {
		return []
	} finally {
		clearTimeout(timer)
	}
}

/**
 * 从 dsh 的凭据文件里取一个命名密钥。
 *
 * 为什么读文件而不是用凭据服务：这个包是**纯 JS、无构建**的宿主插件，注入 `credentials`
 * 服务需要知道它的实例 API；而凭据文件是本应用自己 home 下的一个固定路径，
 * 读它就是确定性的。文件不存在或没有该键时返回空串，由调用方退化。
 *
 * @param file - 凭据文件路径。
 * @param name - 键名，例如 `DEEPSEEK_API_KEY`。
 * @returns 密钥，取不到时为空串。
 */
export function readCredential(file, name) {
	try {
		const text = readFileSync(file, 'utf8')
		const m = new RegExp(`^\\s*${name}\\s*:\\s*(\\S+)\\s*$`, 'm').exec(text)
		return m === null ? '' : m[1].replace(/^['"]|['"]$/g, '')
	} catch {
		return ''
	}
}
