/**
 * 符号白名单校验：把「模型写出来的 API 名字到底存不存在」变成一台机器能回答的问题。
 *
 * 这是**硬**的那一半防幻觉。`search_docs` / `lookup_api` 靠的是模型自觉去查；
 * 这个校验器不需要它自觉 —— 把整段代码丢进来，它直接说哪些名字在官方文档里不存在。
 *
 * ## 移植来源
 *
 * 逐函数移植自教学主线的 `maixrag/evaluation/harness.py`
 * （`extract_symbol_references` / `_symbol_is_valid` / `maix_aliases` /
 * `infer_instance_types`）。那边是评测用的，这边是运行期的；两条路径必须用
 * **同一套判据**，否则评测说"通过"、线上说"幻觉"，就没人知道该信哪个。
 *
 * ## 它抓什么，不抓什么
 *
 * 抓：
 *
 * 1. 全限定名 —— `maix.image.Image()`；
 * 2. 模块别名 —— `from maix import camera` 之后的 `camera.Camera()`；
 * 3. 实例方法 —— 上例之后 `cam = camera.Camera()` 再 `cam.super_zoom()`
 *    （只做**单层构造跟踪**，不做控制流分析：`cam` 被重新赋值、从函数返回、
 *    或作为参数传进来之后，类型就丢了）。
 *
 * 不抓：拼写正确的符号用在错误的地方、参数个数不对、类型不匹配。
 * 这些需要真的跑一遍，静态检查给不了。
 *
 * @module dsh-maixcam-shell/symbols
 */

/** 代码里出现的 maix 全限定名。 */
const SYMBOL_REF_SOURCE = String.raw`\bmaix(?:\.[A-Za-z_]\w*)+`
/** 从自然语言答案里取的代码围栏。 */
const CODE_FENCE_SOURCE = String.raw`\`\`\`[a-zA-Z]*\n([\s\S]*?)\`\`\``
/** `from maix[.sub] import a, b as c`。 */
const IMPORT_FROM_MAIX_SOURCE =
	String.raw`^[ \t]*from[ \t]+maix(?:\.([A-Za-z_][\w.]*))?[ \t]+import[ \t]+([^\n#]+)`
/** `import maix[.sub] [as alias]`。 */
const IMPORT_MAIX_SOURCE =
	String.raw`^[ \t]*import[ \t]+maix(?:\.([A-Za-z_][\w.]*))?(?:[ \t]+as[ \t]+([A-Za-z_]\w*))?`
/** 属性访问链：`alias.Name(.attr)*`。 */
const ATTR_REF_SOURCE = String.raw`\b([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)`
/** 构造赋值：`x = camera.Camera(`。 */
const ASSIGN_CALL_SOURCE = String.raw`([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w.]*)\s*\(`
/** 枚举成员 / 常量：`FMT_RGB888`、`AWB_MODE`。 */
const CONST_MEMBER_SOURCE = String.raw`^[A-Z][A-Z0-9_]*$`

/**
 * 造一个正则。
 *
 * 每次调用都造新的：带 `g` 的正则对象在 JS 里是**有状态**的（`lastIndex`），
 * 复用同一个对象会让第二次扫描从上次停下的位置开始 —— 这类 bug 只在第二次调用时出现，
 * 最难查。
 *
 * @param source - 正则源。
 * @param flags - 附加标志。
 * @returns 新的正则对象。
 */
function re(source, flags = '') {
	return new RegExp(source, flags)
}

/**
 * 要拿去查符号的「代码区域」。
 *
 * 两种调用形态必须分开，而且由调用方说清是哪一种：
 *
 * | 调用方 | 传进来的东西 | 该查哪里 |
 * | --- | --- | --- |
 * | 生成轴评测 | 一整段自然语言答案 | 只查 ``` 围栏里（正文提到一个名字不算断言） |
 * | `check_api_usage` 工具 | 一段裸 Python 源码 | 整段都查（它本来就没有围栏） |
 *
 * 教学主线在这里踩过事故 01（「静默的空结果」）：无条件只找围栏，于是工具对裸代码
 * 永远返回「检查了 0 个符号」，而 0 被当成「没有可校验的符号」返回成功 ——
 * 第一级防幻觉校验在最需要它的那条路径上一次也没生效。
 *
 * @param text - 待检查的文本。
 * @param assumeCode - 是否按「这就是裸代码」处理。
 * @returns 需要检查的代码块数组。
 */
export function codeRegions(text, assumeCode) {
	if (assumeCode) return String(text ?? '').trim() === '' ? [] : [String(text)]
	return [...String(text ?? '').matchAll(re(CODE_FENCE_SOURCE, 'g'))].map((m) => m[1])
}

/**
 * 解析导入语句，返回模块别名与符号别名两张表。
 *
 * 支持四种写法：
 *
 * ```
 * from maix import camera        → modules["camera"] = "maix.camera"
 * from maix.image import Image   → symbols["Image"]  = "maix.image.Image"
 * import maix.image as im        → modules["im"]     = "maix.image"
 * import maix.image              → modules["image"]  = "maix.image"
 * ```
 *
 * 分成两张表是必要的：`from maix import camera` 得到的是**模块**（后面还要接
 * `.Camera`），而 `from maix.image import Image` 得到的是**符号本身**。
 * 混在一起就会把限定名拼错。
 *
 * @param code - 一段代码。
 * @returns `{ modules, symbols }`。
 */
export function maixAliases(code) {
	const modules = new Map()
	const symbols = new Map()

	for (const m of String(code).matchAll(re(IMPORT_FROM_MAIX_SOURCE, 'gm'))) {
		const sub = m[1]
		for (const raw of m[2].split(',')) {
			const parts = raw.trim().split(' as ')
			const name = parts[parts.length - 1].trim()
			if (name === '' || name === '*') continue
			if (sub) symbols.set(name, `maix.${sub}.${name}`)
			else modules.set(name, `maix.${name}`)
		}
	}

	for (const m of String(code).matchAll(re(IMPORT_MAIX_SOURCE, 'gm'))) {
		const sub = m[1]
		const alias = m[2]
		if (sub) modules.set(alias || sub.split('.')[0], `maix.${sub}`)
		else modules.set(alias || 'maix', 'maix')
	}

	return { modules, symbols }
}

/**
 * 把一个被调用名解析成 maix 限定名；不是 maix 的返回 `undefined`。
 *
 * @param callee - 被调用的名字，可能带点。
 * @param modules - 模块别名表。
 * @param symbols - 符号别名表。
 * @returns 限定名或 `undefined`。
 */
function resolveCallee(callee, modules, symbols) {
	if (callee.includes('.')) {
		const [head, ...rest] = callee.split('.')
		const base = modules.get(head) ?? symbols.get(head)
		if (base !== undefined) return `${base}.${rest.join('.')}`
		return callee.startsWith('maix.') ? callee : undefined
	}
	return symbols.get(callee) ?? modules.get(callee)
}

/**
 * 跟踪 `x = camera.Camera(...)` 这类赋值，推断变量所属的 maix 类型。
 *
 * 例程几乎都是
 *
 * ```python
 * from maix import camera
 * cam = camera.Camera()
 * cam.read()          # ← 幻觉最常出现在这里
 * ```
 *
 * `cam` 是局部变量，不理解它的类型就查不出 `cam.super_zoom()` 是编造的。
 * 局限也如实写在这里：**只做单层构造跟踪**，不做控制流分析。
 *
 * @param code - 一段代码。
 * @param modules - 模块别名表。
 * @param symbols - 符号别名表。
 * @returns 变量名 → 限定名。
 */
export function inferInstanceTypes(code, modules, symbols) {
	const types = new Map()
	for (const m of String(code).matchAll(re(ASSIGN_CALL_SOURCE, 'g'))) {
		const qual = resolveCallee(m[2], modules, symbols)
		if (qual !== undefined && qual.startsWith('maix')) types.set(m[1], qual)
	}
	return types
}

/**
 * 抽出文本里引用的 maix 符号，**解析导入别名之后**给出限定名。
 *
 * @param text - 待检查的文本。
 * @param options - 选项。
 * @param options.assumeCode - 是否按「这就是裸代码」处理。
 * @returns 排序去重后的限定名数组。
 */
export function extractSymbolReferences(text, { assumeCode = false } = {}) {
	const refs = new Set()

	for (const block of codeRegions(text, assumeCode)) {
		for (const m of block.matchAll(re(SYMBOL_REF_SOURCE, 'g'))) refs.add(m[0])

		const { modules, symbols } = maixAliases(block)
		const instances = inferInstanceTypes(block, modules, symbols)

		for (const m of block.matchAll(re(ATTR_REF_SOURCE, 'g'))) {
			const base = modules.get(m[1]) ?? instances.get(m[1]) ?? symbols.get(m[1])
			if (base !== undefined) refs.add(base + m[2])
		}

		// `from maix.image import Image` 之后直接 `Image(...)`：没有属性链。
		for (const [name, full] of symbols) {
			if (re(String.raw`\b${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\s*\(`).test(block)) {
				refs.add(full)
			}
		}
	}

	return [...refs].sort()
}

/**
 * 判断一个限定名是否在白名单里。
 *
 * 两条规则，缺一不可：
 *
 * 1. **全名命中** → 有效。
 * 2. **剥离末段后命中，且被剥离的都是枚举成员/常量** → 有效。
 *
 * 第 2 条的必要性：签名里会出现 `maix.image.Format.FMT_RGB888` 这种枚举成员取值，
 * 而白名单可能只登记到枚举类 `maix.image.Format`。不放行会把完全正确的代码判成幻觉。
 *
 * 第 2 条的**边界同样重要**：只允许剥离**常量形**（`FMT_RGB888`、`AWB_MODE`）的段。
 * 否则 `maix.camera.Camera.super_zoom` 会因为前缀 `maix.camera.Camera` 命中而被
 * 误判为合法 —— 而那正是我们要抓的幻觉（教学主线踩过的假阴性）。
 *
 * @param ref - 限定名。
 * @param roster - 白名单集合。
 * @returns 是否有效。
 */
export function symbolIsValid(ref, roster) {
	if (roster.has(ref)) return true

	const parts = ref.split('.')
	for (let end = parts.length - 1; end > 1; end -= 1) {
		const prefix = parts.slice(0, end).join('.')
		if (!roster.has(prefix)) continue
		const suffix = parts.slice(end)
		if (suffix.every((s) => re(CONST_MEMBER_SOURCE).test(s))) return true
		// 前缀命中但后缀不像枚举成员：这是「类存在、但方法不存在」的典型幻觉，
		// 不能放过，继续尝试更短的前缀。
	}
	return false
}

/**
 * 第一级校验：代码里出现的 maix 符号是否都真实存在。
 *
 * @param text - 待检查的文本。
 * @param roster - 白名单集合。
 * @param options - 选项。
 * @param options.assumeCode - 是否按「这就是裸代码」处理。
 * @returns `{ unknown, checked, refs }`；`unknown` 是不存在的符号。
 */
export function checkSymbols(text, roster, { assumeCode = false } = {}) {
	const refs = extractSymbolReferences(text, { assumeCode })
	const unknown = refs.filter((r) => !symbolIsValid(r, roster))
	return { unknown, checked: refs.length, refs }
}
