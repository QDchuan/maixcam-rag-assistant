/**
 * 面向 agent 的工具半边。
 *
 * ## 为什么这几个工具必须存在
 *
 * 这个项目的起点是「模型答得像模像样，但代码是编的」。给模型一个聊天框治不了它 ——
 * 它需要一个**能查到原文、并且必须引用原文**的出口。所以这里把宿主已经加载好的语料
 * 变成两个工具：
 *
 * - `search_docs` —— 按意思检索教程与 API 文档，返回带出处和相似度的片段；
 * - `lookup_api` —— 查一个 API 符号的**精确签名**，用来核对模型写出来的名字。
 *
 * 工具结果刻意渲染成**给人看的文本**而不是一大坨 JSON：默认的工具卡片直接显示
 * 这段文本，用户看到的就是「它到底查到了什么」。这是这个应用相对一个裸聊天框的
 * 全部增量，值得为可读性让路。
 *
 * ## 一个踩过的坑：`parameters` 不是 JSON Schema
 *
 * `defineTool` 的 `parameters` 是一张**扁平的「参数名 → 规格」表**，
 * 必填写在每个属性自己的 `required: true` 上：
 *
 * ```js
 * parameters: { query: { type: 'string', description: '…', required: true } }
 * ```
 *
 * 而不是 JSON Schema 的 `{ type: 'object', properties: {...}, required: [...] }`。
 * 写成后者**不会报错**，只会让这两个工具静默地不出现在模型的工具表里 ——
 * 路由照常工作，界面照常工作，只有 agent 看不见工具。
 * 第一次端到端实测就是这么发现的：模型读了人格里的纪律，然后说
 * 「`search_docs` 不在我的工具列表里」。
 *
 * @module dsh-maixcam-shell/tools
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { checkSymbols } from './symbols.js'

/**
 * 把一段正文压成一行。
 *
 * @param text - 原始文本。
 * @param max - 最多留几个字。
 * @returns 压平后的摘要。
 */
function oneLine(text, max) {
	const flat = String(text ?? '').replace(/\s+/g, ' ').trim()
	return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

/**
 * 把检索结果渲染成模型与人都好读的文本。
 *
 * 每条都带**出处**与**相似度**：相似度让模型能看出「这条其实不太像」，
 * 出处让用户能自己点过去核对。两者缺一，这个工具就退化成一个更慢的搜索框。
 *
 * 除命中之外还交一份**模块覆盖度**：命中落在哪几个模块、语料一共几个模块、
 * 哪些模块**一条都没命中**。这一行是给模型看的「你这次只覆盖了这么窄」——
 * 实测的病正是它在 `projects`/`vision` 里找到一页就当成全部，
 * 而没有任何东西告诉它「PWM、UART 这些模块你一条都没碰到」。
 *
 * @param result - `/search` 返回的结构。
 * @returns 模型面向的内容块。
 */
function renderHits(result) {
	// ── 多面向：逐个子系统报命中与**缺口** ──────────────────────────────────
	if (Array.isArray(result.facets)) {
		const out = [
			`多面向检索：${result.facets.length} 个子系统（${result.strategy} · 共 ${result.totalMs} ms）`,
			'',
		]
		for (const f of result.facets) {
			if (f.n === 0) {
				out.push(`## ${f.aspect} —— **本地无证据**`)
				out.push('    本地知识库里没有关于这一块的内容。必须在回答里明说，不要凭印象补全。')
			} else {
				const mods = f.coverage.hit.map((h) => `${h.module}(${h.n})`).join(' ')
				out.push(`## ${f.aspect} —— 命中 ${f.n} 条（模块 ${mods}）`)
				for (const [i, h] of f.hits.entries()) {
					const where = [h.doc_title || h.doc_id, ...(h.heading_path ?? [])]
						.filter((s) => s && s !== '(开头)').join(' › ')
					out.push(`  [${i + 1}] ${h.score.toFixed(3)} ${where}`)
					if (h.url) out.push(`      来源：${h.url}`)
					out.push(`      片段：${oneLine(h.text, 200)}`)
				}
			}
			out.push('')
		}
		if (result.empty.length > 0) {
			out.push(`**有 ${result.empty.length} 个子系统本地无证据：${result.empty.join(' · ')}**`)
			out.push('这些就是这次回答的边界。不要用看起来合理的方案把它们填上。')
		}
		out.push('说明：回答时只使用上面出现的内容并写明来源；缺口要明说。')
		return [{ type: 'text', text: out.join('\n') }]
	}

	const head =
		`知识库命中 ${result.hits.length} 条（${result.strategy} · 嵌入 ${result.embedMs} ms · 共 ${result.totalMs} ms）`

	const cov = result.coverage
	const covLines = []
	if (cov) {
		covLines.push(
			`本次命中覆盖 ${cov.hit.length} / ${cov.total} 个模块：`
				+ cov.hit.map((h) => `${h.module}(${h.n})`).join(' · '),
		)
	}

	// ── 前置闸门：设计题不许拿单查询的结果作答 ────────────────────────────
	//
	// 实测过的病：问「设计一个人脸跟随系统」，单查询的 6 条命中**全部**落在
	// `projects` / `vision` 两个模块里，模型就把那一页当成了全部，从没想到还要看
	// 舵机、PWM、引脚、供电 —— 而那些内容确实在语料里，只是在别的模块。
	//
	// 关联本身是**模型**能做而**向量**做不到的事（实测：模型自己打出过
	// 「总线舵机 串口舵机 舵机供电 外部电源 电压」这种扩展）。所以这里不自己猜
	// 扩展词，而是把「你必须先扩展」变成一道过不去的门：设计类问题不给
	// `aspects`，就不让它拿到一份看起来完整的答案。
	if (result.needsAspects === true) {
		covLines.push(
			'',
			'## ⚠️ 这是一道设计类问题，而你只查了一次',
			'单条查询只会返回**最像整题的那一页**，它天然覆盖不到这个系统要用到的其他部件。',
			`这一查只碰到 ${cov ? cov.hit.length : '?'} 个模块 —— 上面的结果**不构成**可以据此设计的依据。`,
			'',
			'**现在停下来做这件事**：把这个系统拆成工程子系统'
				+ '（例如人脸跟随系统 → 人脸检测、舵机控制、串口/总线通信、供电与电流、机械结构、引脚分配），'
				+ '把清单作为 `aspects` 重新调用 `search_docs`。'
				+ '工具会逐个子系统告诉你**本地有没有证据**，没有证据的那些子系统就是你这次回答的边界。',
			'不要用这一查的结果直接给设计方案。',
		)
	}

	const lines = result.hits.map((hit, i) => {
		const where = [hit.doc_title || hit.doc_id, ...(hit.heading_path ?? [])]
			.filter((s) => s && s !== '(开头)')
			.join(' › ')
		const from = hit.from
			? `（稠密 #${hit.from.dense ?? '—'} · 稀疏 #${hit.from.sparse ?? '—'}）`
			: ''
		return [
			`[${i + 1}] score=${hit.score.toFixed(3)} ${from} ${where}`,
			hit.url ? `    来源：${hit.url}` : null,
			`    片段：${oneLine(hit.text, 260)}`,
		]
			.filter((s) => s !== null)
			.join('\n')
	})
	const tail =
		'说明：以上片段是本次检索能给出的全部依据。回答时只使用这里出现的内容，'
		+ '并写出对应的来源；没有覆盖到的部分直说不知道，不要补全。'
	return [{ type: 'text', text: [head, ...covLines, '', ...lines, '', tail].join('\n') }]
}

/**
 * 把符号校验的结果渲染成给人读的文本。
 *
 * 三种结局必须**在文字上分得开**，因为它们对模型意味着完全不同的下一步：
 *
 * - 全部存在 → 可以交付；
 * - 发现不存在的符号 → 去 `lookup_api` 查真的，或删掉；
 * - **校验器自己失效**（一段明显是 MaixPy 的代码里一个符号都没解析出来）
 *   → 这一关**没有通过**，不许当成"没问题"。
 *
 * 第三种是教学主线的事故 01：一个静默放行的防幻觉校验，比没有校验更危险 ——
 * 它给了模型和人一个虚假的安心。
 *
 * @param result - `{ unknown, checked }`。
 * @param code - 被检查的代码，用来判断"这是不是一段 MaixPy 代码"。
 * @returns 内容块。
 */
function renderCheck(result, code) {
	const { unknown, checked } = result
	if (checked === 0) {
		const looksLikeMaix = /maix|gpio|camera/i.test(code)
		if (looksLikeMaix) {
			return [
				{
					type: 'text',
					text:
						'第一级校验没能从这段代码里解析出任何 maix 符号，但它看起来确实在用 MaixPy。'
						+ '**这表示校验器失效了，不等于代码通过。**'
						+ '请改用 `lookup_api` 逐个确认符号。',
				},
			]
		}
		return [
			{
				type: 'text',
				text: '代码里没有出现可校验的 maix 符号（没有 `maix.*` 引用，也没有从 maix 导入后的调用）。',
			},
		]
	}
	if (unknown.length === 0) {
		return [{ type: 'text', text: `检查了 ${checked} 个符号，全部存在于官方 API 名单。` }]
	}
	const lines = [`检查了 ${checked} 个符号，发现 ${unknown.length} 个不存在：`]
	for (const u of unknown) lines.push(`  · ${u}`)
	lines.push('')
	lines.push(
		'**这些符号在官方文档里不存在，属于编造。**'
			+ '请用 `lookup_api` 查到真实 API 再改，或删除相关代码；不要把名字改得像一点就交。',
	)
	return [{ type: 'text', text: lines.join('\n') }]
}

/**
 * 造出这一组工具。
 *
 * @param deps - 依赖。
 * @param {() => Promise<object>} deps.getCorpus - 取语料句柄（惰性加载）。
 * @param {(text: string, k: number) => Promise<object>} deps.search - 跑一次检索。
 * @param {(name: string) => Promise<object>} deps.lookup - 查一个符号。
 * @param {(code: string) => Promise<object>} deps.check - 校验一段代码里的符号。
 * @param {() => number} deps.getTimeoutMs - 单次调用超时。
 * @returns 工具定义数组。
 */
export function createCorpusTools(deps) {
	return [
		defineTool({
			name: 'search_docs',
			description:
				'在本地 MaixPy / MaixCAM 教程与 API 文档里做语义检索，返回带出处与相似度的原文片段。'
				+ '回答任何关于 MaixCAM / MaixPy 的问题（怎么写代码、某个功能怎么做、报错怎么排查）之前都要先调用它，'
				+ '并以返回的片段作为唯一依据；没有命中就直说不知道，不要凭记忆补全。',
			parameters: {
				query: {
					type: 'string',
					description: '要检索的自然语言问题或关键词，例如「怎么寻找色块」。',
					required: true,
				},
				k: {
					type: 'integer',
					description: '每个面向取几条，默认 6，最多 20。',
				},
				aspects: {
					type: 'array',
					items: { type: 'string' },
					description:
						'子系统清单。**问「怎么设计/实现某个系统」时必须给**：先把系统拆成若干子系统'
						+ '（例如人脸跟随系统 → 人脸检测、舵机控制、串口通信、供电、机械结构），'
						+ '每个子系统一个字符串。工具会逐个检索并逐个告诉你哪个子系统本地没有证据 —— '
						+ '那些就是你这次回答的边界。简单的事实性问题可以不给。',
				},
			},
			output: { schema: { type: 'json' }, render: (_args, value) => renderHits(value) },
			timeoutMs: deps.getTimeoutMs(),
			isConcurrencySafe: () => true,
			async execute(args) {
				const k = Math.min(Math.max(Number(args.k) || 6, 1), 20)
				const aspects = Array.isArray(args.aspects)
					? args.aspects.map((s) => String(s ?? '').trim()).filter((s) => s !== '').slice(0, 10)
					: []
				return deps.search(String(args.query ?? ''), k, aspects)
			},
		}),

		defineTool({
			name: 'lookup_api',
			description:
				'查一个 MaixPy API 符号的精确签名与所属模块（支持 `camera.AeMode` 这样的限定名，也支持裸名）。'
				+ '写代码用到某个 API 之前先查一下，用它来核对拼写与参数；返回「没找到」时不要凭印象写出这个名字。',
			parameters: {
				name: {
					type: 'string',
					description: '符号名，限定名或裸名，例如 `camera.AeMode` 或 `find_blobs`。',
					required: true,
				},
			},
			output: {
				schema: { type: 'json' },
				render: (_args, value) => {
					if (value.found !== true) {
						return [
							{
								type: 'text',
								text: `没有找到符号 \`${value.name}\`。它不在这份 API 名单里 —— 不要凭印象使用这个名字。`,
							},
						]
					}
					const head =
						value.exact !== null
							? `\`${value.exact.qualname}\`（${value.exact.kind}）`
							: `\`${value.name}\` 不是限定名，下面是同名符号：`
					const rows = value.exact !== null ? [value.exact] : value.candidates
					const lines = rows.map((s) =>
						[
							`- ${s.qualname}  [${s.kind}]  module=${s.module}`,
							s.signature ? `  签名：${s.signature}` : null,
							s.summary ? `  说明：${oneLine(s.summary, 140)}` : null,
							s.url ? `  来源：${s.url}` : null,
						]
							.filter((x) => x !== null)
							.join('\n'),
					)
					const rosterNote =
						value.inRoster === true
							? ''
							: '\n注意：这个符号不在白名单里，使用前请再确认。'
					return [{ type: 'text', text: [head, ...lines, rosterNote].join('\n') }]
				},
			},
			timeoutMs: deps.getTimeoutMs(),
			isConcurrencySafe: () => true,
			async execute(args) {
				return deps.lookup(String(args.name ?? ''))
			},
		}),

		defineTool({
			name: 'check_api_usage',
			description:
				'把一整段 MaixPy 代码丢进来，校验里面出现的 maix 符号是否都真实存在。'
				+ '它会解析导入别名（`from maix import camera` 之后的 `camera.Camera()`）和单层构造赋值'
				+ '（`cam = camera.Camera()` 之后的 `cam.read()`），所以能抓住编造模块成员与编造实例方法'
				+ '这两类最常见的幻觉。**给用户交付代码之前必须调用它**；返回"发现不存在的符号"时，'
				+ '不要改个更像的名字就交，要先用 `lookup_api` 查到真的。',
			parameters: {
				code: {
					type: 'string',
					description: '要校验的 MaixPy 代码（整段源码，不需要 markdown 围栏）。',
					required: true,
				},
			},
			output: {
				schema: { type: 'json' },
				render: (_args, value) => renderCheck(value, String(_args.code ?? '')),
			},
			timeoutMs: deps.getTimeoutMs(),
			isConcurrencySafe: () => true,
			async execute(args) {
				const code = String(args.code ?? '')
				if (code.trim() === '') {
					throw new Error('code 不能为空。')
				}
				return deps.check(code)
			},
		}),
	]
}

/**
 * 语料没就绪时，工具要说的那句话。
 *
 * 工具失败**必须说清原因**：模型收到「检索不可用」会去说不知道，收到一个空结果
 * 却会以为「知识库里没有」，然后继续凭记忆回答 —— 那正是要治的病。
 *
 * @param error - 捕获到的异常。
 * @returns 原因字符串。
 */
export function unavailable(error) {
	return `知识库不可用，本次检索没有执行：${error?.message ?? String(error)}。`
		+ '请告诉用户知识库没就绪，不要凭记忆回答。'
}
