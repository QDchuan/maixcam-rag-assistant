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
 *
 * 三个都是 `single` 座位，登记即替换。`ctx.slots.inject(key, cb)` 的意义在于
 * **不必关心声明者先加载还是后加载**：声明一出现，回调就跑；声明消失，登记
 * 连同它一起撤回。
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
		 * 需要真 CSS 的部分（字体、字距、过渡）。放在内联样式里能画出来的东西
		 * 一律留在内联样式里，这里只放内联样式表达不了的那些。
		 */
		const CSS = [
			'.dsh-mx-name{display:inline-flex;align-items:baseline;gap:6px;min-width:0;',
			'font-size:13px;font-weight:600;letter-spacing:.2px;white-space:nowrap;',
			'overflow:hidden;text-overflow:ellipsis}',
			'.dsh-mx-name-tag{flex:0 0 auto;padding:1px 5px;border-radius:5px;',
			'border:1px solid currentColor;font-size:9px;font-weight:500;line-height:14px;',
			'letter-spacing:.4px;opacity:.62}',
			'.dsh-mx-mark{display:block;flex:0 0 auto}',
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
		}

		exports.apply = apply
		exports.inject = inject
		return module.exports
	},
})
