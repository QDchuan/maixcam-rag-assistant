/**
 * `dsh-maixcam-shell` 的浏览器半边 —— 这一份文件就是「壳」。
 *
 * ## 为什么是手写的 bundle
 *
 * 它不是打包产物，而是直接用宿主自己的 lazy-CJS bundle 格式写成的
 * （`window.__ModuleLoader__.load({ id, factory })`）。因此这个包**不需要任何
 * 构建步骤**：宿主把这份文件原样当作包的 `./client` 导出交给浏览器。
 * 少一条构建链，就少一处「改了没生效 / 忘了重新构建」的坑。
 *
 * ## 它占哪些座位
 *
 * 界面在 Harness 里是「洞（slot）+ 占位者（occupant）」。壳体不重写布局，
 * 只是往既有座位上放自己的东西：
 *
 * - `sidebar.brand.mark` —— 侧栏的品牌图标（展开态与折叠轨道共用）。
 * - `sidebar.brand.name` —— 侧栏品牌名。
 * - `conversation.hero.brand.mark` —— 空白会话首屏标题前的品牌图标。
 * - `sidebar.panellist` + `main` —— **MaixCAM 知识库面板**。侧栏一个图标按钮，
 *   中央一块面板，两者用同一个 id 配对。这是这个壳里唯一「有内容」的地方：
 *   数据来自宿主路由 `/api/maixcam/*`，把「问题 → 命中了哪几条切片 → 多少分 →
 *   来自哪一页」摊开给人看。
 *
 * 品牌那三处都是 `single` 座位，登记即替换；`sidebar.panellist` 是 `list`，
 * `main` 是 `keyed`，登记即新增。两种语义的差别在 {@link TAKE_OVER} 上写清楚了。
 *
 * ## 依赖只有 react
 *
 * 只 `require('react')`，其余一律用 `React.createElement` 与内联样式。
 * 外壳的静态 UI 基座（`@deepseek-ai/dsh-client-ui-primitives`）当然可以引，
 * 但引一个就多一处版本耦合；这一版刻意只依赖最稳的那一个。
 *
 * @module dsh-maixcam-shell/client
 */

window.__ModuleLoader__.load({
	id: 'dsh-maixcam-shell',
	factory: (require) => {
		const module = { exports: {} }
		const exports = module.exports
		Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })

		const React = require('react')
		const h = React.createElement

		/** 品牌名：界面上看到的产品名，不是底座的名字。 */
		const PRODUCT_NAME = 'MaixCAM 开发助手'
		/** 样式表标记，保证同一份样式只注入一次。 */
		const STYLE_MARK = 'dsh-maixcam-shell'

		/**
		 * 抢占座位用的优先级。
		 *
		 * 这一条是本文件里最不直观、也最容易踩的地方，值得写清楚：
		 *
		 * `single` 座位的选举规则是「按优先级排序后，**第一个**存活的条目获胜」。
		 * 也就是说**数字越小越优先**。而壳自己的默认占位者（侧栏的鱼形标志与
		 * 本地构建标签）就是一个 `priority: 0` 的普通占位者 —— 它不是「没有占位者
		 * 时的兜底」，而是一个真真正正在册的条目。
		 *
		 * 所以「填一个座位」和「换掉一个座位」是两件事：
		 *
		 * - `conversation.hero.brand.mark` 原本没人占，登记就赢；
		 * - `sidebar.brand.mark` / `sidebar.brand.name` 已经有占位者，
		 *   不给优先级就是并列 0，先登记的默认占位者赢，你的组件根本不渲染 ——
		 *   而且**不报错**，页面看起来「什么都没发生」。
		 *
		 * 实测确认：`priority: -1` 赢，`priority: +1` 输。
		 */
		const TAKE_OVER = -1

		/**
		 * 需要真 CSS 的部分（字体、字距、过渡、滚动、悬停）。放在内联样式里能画出来的
		 * 东西一律留在内联样式里，这里只放内联样式表达不了的那些。
		 */
		const CSS = [
			'.dsh-mx-name{display:inline-flex;align-items:baseline;gap:6px;min-width:0;',
			'font-size:13px;font-weight:600;letter-spacing:.2px;white-space:nowrap;',
			'overflow:hidden;text-overflow:ellipsis}',
			'.dsh-mx-name-tag{flex:0 0 auto;padding:1px 5px;border-radius:5px;',
			'border:1px solid currentColor;font-size:9px;font-weight:500;line-height:14px;',
			'letter-spacing:.4px;opacity:.62}',
			'.dsh-mx-mark{display:block;flex:0 0 auto}',
			'.dsh-mx-kb{display:flex;flex-direction:column;gap:14px;height:100%;box-sizing:border-box;',
			'padding:18px 22px;overflow:auto;font-size:13px;line-height:20px}',
			'.dsh-mx-kb h2{margin:0;font-size:15px;font-weight:600}',
			'.dsh-mx-kb-chips{display:flex;flex-wrap:wrap;gap:6px}',
			'.dsh-mx-chip{padding:2px 8px;border-radius:999px;border:1px solid rgba(127,127,127,.3);',
			'font-size:11px;line-height:16px;opacity:.8;white-space:nowrap}',
			'.dsh-mx-chip[data-tone="bad"]{border-color:var(--dsw-alias-state-error-primary,#e5534b);',
			'color:var(--dsw-alias-state-error-primary,#e5534b);opacity:1}',
			'.dsh-mx-kb-row{display:flex;gap:8px}',
			'.dsh-mx-kb-input{flex:1 1 auto;min-width:0;box-sizing:border-box;padding:7px 11px;',
			'border-radius:8px;border:1px solid rgba(127,127,127,.34);background:transparent;',
			'color:inherit;font:inherit;font-size:13px}',
			'.dsh-mx-kb-input:focus{outline:none;border-color:rgba(127,127,127,.7)}',
			'.dsh-mx-kb-btn{flex:0 0 auto;padding:7px 16px;border-radius:8px;cursor:pointer;',
			'border:1px solid rgba(127,127,127,.42);background:rgba(127,127,127,.14);',
			'color:inherit;font:inherit;font-size:13px}',
			'.dsh-mx-kb-btn:hover:not(:disabled){background:rgba(127,127,127,.24)}',
			'.dsh-mx-kb-btn:disabled{cursor:default;opacity:.5}',
			'.dsh-mx-hit{display:flex;flex-direction:column;gap:5px;padding:11px 13px;border-radius:10px;',
			'border:1px solid rgba(127,127,127,.24);background:rgba(127,127,127,.05)}',
			'.dsh-mx-hit-top{display:flex;align-items:baseline;gap:8px;min-width:0}',
			'.dsh-mx-rank{flex:0 0 auto;width:17px;height:17px;border-radius:5px;font-size:10px;',
			'line-height:17px;text-align:center;background:rgba(127,127,127,.2)}',
			'.dsh-mx-hit-doc{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
			'.dsh-mx-hit-path{font-size:11px;opacity:.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
			// 出处链接：不写颜色就会落到浏览器默认蓝（实测 rgb(0,0,238)），在深色底上很刺眼。
			// 用主题的链接令牌，明暗两套都跟着走。
			'.dsh-mx-hit a{color:var(--dsw-alias-link,#4de2ff);text-decoration-color:rgba(0,229,255,.4)}',
			'.dsh-mx-hit a:hover{text-decoration-color:currentColor}',
			'.dsh-mx-badge{flex:0 0 auto;padding:1px 6px;border-radius:5px;font-size:10px;line-height:15px;',
			'border:1px solid rgba(127,127,127,.34);opacity:.75}',
			'.dsh-mx-score{margin-left:auto;flex:0 0 auto;font-size:11px;opacity:.6;',
			'font-variant-numeric:tabular-nums}',
			'.dsh-mx-hit-text{white-space:pre-wrap;word-break:break-word;max-height:96px;overflow:hidden;',
			'font-size:12px;line-height:18px;opacity:.82}',
			'.dsh-mx-note{opacity:.6;font-size:12px}',
			'.dsh-mx-warn{padding:10px 13px;border-radius:9px;font-size:12px;line-height:18px;',
			'border:1px solid var(--dsw-alias-state-error-primary,#e5534b);',
			'color:var(--dsw-alias-state-error-primary,#e5534b)}',
		].join('')


		/**
		 * 把样式表挂到 `<head>` 上，返回撤销函数。
		 *
		 * 先查再插，是因为同一个页面里可能有多份来源都在尝试注入；用 `data-`
		 * 标记认领，而不是靠调用次数。
		 *
		 * @returns 移除本插件所注入样式表的函数。
		 */
		function installStyles() {
			const existing = document.querySelector(`style[data-${STYLE_MARK}]`)
			if (existing !== null) return () => {}

			const el = document.createElement('style')
			el.setAttribute(`data-${STYLE_MARK}`, '')
			el.textContent = CSS
			document.head.appendChild(el)
			return () => el.remove()
		}

		/**
		 * MaixCAM 的品牌图标：一个摄像头模组 —— 圆角机身、镜头、一颗指示灯。
		 *
		 * 用 `currentColor` 绘制，所以它自动跟随明暗主题，不需要任何主题令牌。
		 *
		 * @param props - 宿主给的几何信息。
		 * @param props.size - 请求的方形边长（像素）。
		 * @returns 品牌图标元素。
		 */
		function MaixMark({ size = 24 }) {
			return h(
				'svg',
				{
					className: 'dsh-mx-mark',
					width: size,
					height: size,
					viewBox: '0 0 24 24',
					fill: 'none',
					role: 'img',
					'aria-label': PRODUCT_NAME,
					focusable: 'false',
				},
				h('rect', {
					x: 2.6,
					y: 2.6,
					width: 18.8,
					height: 18.8,
					rx: 5.4,
					stroke: 'currentColor',
					strokeWidth: 1.5,
					opacity: 0.5,
				}),
				h('circle', {
					cx: 12,
					cy: 12.4,
					r: 5.1,
					stroke: 'currentColor',
					strokeWidth: 1.5,
				}),
				h('circle', { cx: 12, cy: 12.4, r: 2, fill: 'currentColor' }),
				h('circle', { cx: 16.9, cy: 7.2, r: 1.05, fill: 'currentColor', opacity: 0.7 }),
			)
		}

		/**
		 * 侧栏品牌名，带一个小标签说明「这是 MaixCAM 助手」而不是通用 agent。
		 *
		 * @returns 品牌名元素。
		 */
		function MaixBrandName() {
			return h(
				'span',
				{ className: 'dsh-mx-name', title: PRODUCT_NAME },
				h('span', null, PRODUCT_NAME),
				h('span', { className: 'dsh-mx-name-tag' }, 'MaixPy'),
			)
		}

		// ── 知识库面板 ───────────────────────────────────────────────────────────
		//
		// 这个面板是这个壳里唯一「有内容」的地方，也是它值得存在的原因：
		// 这个项目的起点是「模型答得像模像样但是编的」，而编造的根源是**看不见证据**。
		// 所以这里把「问题 → 命中了哪几条切片 → 各自多少分 → 来自哪一页」直接摊开。
		//
		// 数据全部来自宿主路由 `/api/maixcam/*`，浏览器不碰语料文件、也不知道
		// 嵌入服务的地址。

		/** 面板 id。侧栏图标与主面板用同一个 id 配对。 */
		const PANEL_ID = 'maixcam-kb'
		/** 面板标题：侧栏按钮的标签与面板标题共用一份文案。 */
		const PANEL_TITLE = 'MaixCAM 知识库'
		/** 宿主路由前缀。 */
		const API = '/api/maixcam'
		/** 一次取几条证据。 */
		const PAGE_SIZE = 6
		/** 空态给的几个起手问题：让人知道「这个库能问什么」。 */
		const EXAMPLES = ['怎么寻找色块', '摄像头怎么初始化', '串口通信怎么用', '人脸检测']

		/**
		 * 把一段正文压成一行摘要。
		 *
		 * @param text - 原始文本。
		 * @param max - 最多留几个字。
		 * @returns 压平后的摘要。
		 */
		function summarize(text, max) {
			const flat = String(text ?? '').replace(/\s+/g, ' ').trim()
			return flat.length > max ? `${flat.slice(0, max)}…` : flat
		}

		/**
		 * 知识库的图标：一枚芯片 —— 方体加四边引脚，一眼能认出是「设备知识」。
		 *
		 * @param props - 宿主给的几何信息。
		 * @param props.size - 请求的方形边长。
		 * @param props.active - 该面板是否正被选中。
		 * @returns 图标元素。
		 */
		function MaixKbIcon({ size = 18, active = false }) {
			const pins = []
			for (const t of [9.2, 12, 14.8]) {
				pins.push(
					h('line', { key: `t${t}`, x1: t, y1: 3.4, x2: t, y2: 6.6, stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }),
					h('line', { key: `b${t}`, x1: t, y1: 17.4, x2: t, y2: 20.6, stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }),
					h('line', { key: `l${t}`, x1: 3.4, y1: t, x2: 6.6, y2: t, stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }),
					h('line', { key: `r${t}`, x1: 17.4, y1: t, x2: 20.6, y2: t, stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }),
				)
			}
			return h(
				'svg',
				{
					className: 'dsh-mx-mark',
					width: size,
					height: size,
					viewBox: '0 0 24 24',
					fill: 'none',
					'aria-hidden': 'true',
					focusable: 'false',
					style: { opacity: active ? 1 : 0.72 },
				},
				h('rect', { x: 6.6, y: 6.6, width: 10.8, height: 10.8, rx: 2.6, stroke: 'currentColor', strokeWidth: 1.6 }),
				h('rect', { x: 10.1, y: 10.1, width: 3.8, height: 3.8, rx: 1, fill: 'currentColor', opacity: active ? 0.95 : 0.6 }),
				pins,
			)
		}

		/**
		 * 一条证据卡片。
		 *
		 * 三条信息缺一不可：**来自哪一页**（可核验）、**哪一段标题下**（可定位）、
		 * **多少分**（可比较）。这也是这个壳相对「一个聊天框」的全部增量。
		 *
		 * @param props - 组件属性。
		 * @param props.hit - 宿主返回的一条证据。
		 * @param props.index - 从 0 开始的排名。
		 * @returns 卡片元素。
		 */
		function EvidenceCard({ hit, index }) {
			return h(
				'div',
				{ className: 'dsh-mx-hit' },
				h(
					'div',
					{ className: 'dsh-mx-hit-top' },
					h('span', { className: 'dsh-mx-rank' }, String(index + 1)),
					h('span', { className: 'dsh-mx-hit-doc' }, hit.doc_title || hit.doc_id),
					h('span', { className: 'dsh-mx-badge' }, hit.kind),
					h('span', { className: 'dsh-mx-score' }, hit.score.toFixed(3)),
				),
				hit.heading_path && hit.heading_path.length > 0
					? h('div', { className: 'dsh-mx-hit-path' }, hit.heading_path.join(' › '))
					: null,
				h('div', { className: 'dsh-mx-hit-text' }, summarize(hit.text, 220)),
				hit.url
					? h(
							'a',
							{
								className: 'dsh-mx-hit-path',
								href: hit.url,
								target: '_blank',
								rel: 'noreferrer',
								title: hit.url,
							},
							summarize(hit.url, 72),
						)
					: null,
			)
		}

		/**
		 * 知识库面板：语料状态 + 一句检索 + 带出处的命中列表。
		 *
		 * @returns 面板元素。
		 */
		function KnowledgePanel() {
			const [status, setStatus] = React.useState(null)
			const [query, setQuery] = React.useState('')
			const [result, setResult] = React.useState(null)
			const [busy, setBusy] = React.useState(false)
			const [error, setError] = React.useState('')

			// 挂载时读一次状态：面板要先回答「你现在到底知道什么」。
			React.useEffect(() => {
				let alive = true
				fetch(`${API}/status`)
					.then((r) => r.json())
					.then((s) => {
						if (alive) setStatus(s)
					})
					.catch((e) => {
						if (alive) setError(`读不到语料状态：${e.message}`)
					})
				return () => {
					alive = false
				}
			}, [])

			/**
			 * 跑一次检索。
			 *
			 * @param text - 查询词。
			 * @returns 完成后的 promise。
			 */
			const run = React.useCallback(
				async (text) => {
					const q = String(text ?? '').trim()
					if (q === '') return
					setBusy(true)
					setError('')
					try {
						const res = await fetch(
							`${API}/search?q=${encodeURIComponent(q)}&k=${PAGE_SIZE}`,
						)
						const body = await res.json()
						if (body.ok) {
							setResult(body)
						} else {
							// 失败与「没命中」必须分开报：把前者显示成后者，
							// 正是这个项目要治的那个病。
							setResult(null)
							setError(body.error ?? `检索失败（HTTP ${res.status}）`)
						}
					} catch (e) {
						setResult(null)
						setError(`检索请求失败：${e.message}`)
					} finally {
						setBusy(false)
					}
				},
				[],
			)

			const stats = status && status.stats ? status.stats : null
			const broken = status && status.loaded === false

			const chips = []
			if (stats) {
				chips.push(`切片 ${stats.chunks}`)
				chips.push(`API 符号 ${stats.symbols}`)
				chips.push(`名单 ${stats.roster}`)
				chips.push(stats.embeddingModel)
				if (stats.corpusFingerprint) {
					chips.push(`指纹 ${stats.corpusFingerprint.slice(0, 8)}`)
				}
			}

			return h(
				'div',
				{ className: 'dsh-mx-kb' },
				h('h2', null, 'MaixCAM 知识库'),
				h(
					'div',
					{ className: 'dsh-mx-note' },
					'检索本地 MaixPy / MaixCAM 教程与 API 文档，每条结果都带出处和相似度 —— ',
					'用来核对模型的回答，而不是相信它。',
				),

				broken
					? h('div', { className: 'dsh-mx-warn' }, `语料没就绪：${status.error}`)
					: chips.length > 0
						? h(
								'div',
								{ className: 'dsh-mx-kb-chips' },
								chips.map((c) => h('span', { className: 'dsh-mx-chip', key: c }, c)),
							)
						: h('div', { className: 'dsh-mx-note' }, '正在读取语料状态…'),

				h(
					'div',
					{ className: 'dsh-mx-kb-row' },
					h('input', {
						className: 'dsh-mx-kb-input',
						type: 'text',
						value: query,
						placeholder: '问一句，例如「怎么寻找色块」',
						onChange: (e) => setQuery(e.target.value),
						onKeyDown: (e) => {
							if (e.key === 'Enter') run(query)
						},
					}),
					h(
						'button',
						{
							className: 'dsh-mx-kb-btn',
							type: 'button',
							disabled: busy || query.trim() === '',
							onClick: () => run(query),
						},
						busy ? '检索中…' : '检索',
					),
				),

				error !== '' ? h('div', { className: 'dsh-mx-warn' }, error) : null,

				result
					? h(
							'div',
							{ className: 'dsh-mx-note' },
							`命中 ${result.hits.length} 条 · 稠密检索 · 嵌入 ${result.embedMs} ms · `,
							`共 ${result.totalMs} ms`,
						)
					: null,

				result && result.hits.length > 0
					? h(
							'div',
							{ style: { display: 'flex', flexDirection: 'column', gap: '8px' } },
							result.hits.map((hit, i) =>
								h(EvidenceCard, { key: hit.chunk_id, hit, index: i }),
							),
						)
					: null,

				!result && error === ''
					? h(
							'div',
							{ className: 'dsh-mx-kb-chips' },
							EXAMPLES.map((ex) =>
								h(
									'button',
									{
										key: ex,
										className: 'dsh-mx-chip',
										type: 'button',
										style: { cursor: 'pointer', font: 'inherit' },
										onClick: () => {
											setQuery(ex)
											run(ex)
										},
									},
									ex,
								),
							),
						)
					: null,
			)
		}

		/** Cordis 服务：只需要槽位注册表。 */
		const inject = ['slots']
		/**
		 * 占住三个品牌座位。
		 *
		 * `sidebar.brand.mark` 与 `sidebar.brand.name` 用嵌套 `inject` 组成一次
		 * 登记集合：两个座位由同一个声明者给出，一起出现也一起消失，避免只换掉
		 * 图标却留着旧名字的半截状态。
		 *
		 * 三处都带 {@link TAKE_OVER}：前两处是在换掉壳的默认品牌，第三处虽然原本
		 * 没人占，也一并显式压过 —— 免得将来底座给首屏加了兜底占位者，这里又静默失效。
		 *
		 * @param ctx - 客户端根上下文。
		 */
		function apply(ctx) {
			ctx.effect(installStyles, 'dsh-maixcam-shell: styles')

			ctx.slots.inject('sidebar.brand.mark', () =>
				ctx.slots.inject('sidebar.brand.name', function* () {
					yield ctx.slots.register(
						{ name: 'sidebar.brand.mark', priority: TAKE_OVER },
						MaixMark,
					)
					yield ctx.slots.register(
						{ name: 'sidebar.brand.name', priority: TAKE_OVER },
						MaixBrandName,
					)
				}),
			)

			ctx.slots.inject('conversation.hero.brand.mark', () =>
				ctx.slots.register(
					{ name: 'conversation.hero.brand.mark', priority: TAKE_OVER },
					MaixMark,
				),
			)

			// 知识库面板：侧栏一个图标 + 中央一块面板，两者用同一个 id 配对。
			// `main` 是 keyed 座位，除保留键 `conversation` 之外的键都拿到
			// 「没有 Session 绑定」的一块地方 —— 正好适合一个与对话无关的库。
			ctx.slots.inject('main', () =>
				ctx.slots.inject('sidebar.panellist', function* () {
					yield ctx.slots.register(
						{ name: 'sidebar.panellist', id: PANEL_ID, order: 20, label: () => PANEL_TITLE },
						MaixKbIcon,
					)
					yield ctx.slots.register({ name: 'main', key: PANEL_ID }, KnowledgePanel)
				}),
			)
		}

		exports.apply = apply
		exports.inject = inject
		return module.exports
	},
})
