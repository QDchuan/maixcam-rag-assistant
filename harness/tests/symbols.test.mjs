/**
 * 符号校验器的回归测试。
 *
 * 这不是「顺手跑一下」的冒烟测试，它守的是一条具体的界线：
 * **运行期的 JS 校验器必须和教学主线的 Python 校验器给同一个答案。**
 * 两条路径判据不一致的话，评测说「通过」、线上说「幻觉」，就没人知道该信哪个。
 *
 * 表里每一条都对应一个真实踩过的坑，注释写清楚是哪一个。
 *
 * 跑法（在仓库根目录）：
 *
 * ```bash
 * node harness/tests/symbols.test.mjs
 * ```
 */

import { readFileSync } from 'node:fs'
import { checkSymbols, extractSymbolReferences, symbolIsValid } from '../dsh-maixcam-shell/lib/symbols.js'

const ROSTER_PATH = new URL('../../corpus/processed/api_roster.txt', import.meta.url)
const roster = new Set(
	readFileSync(ROSTER_PATH, 'utf8')
		.split('\n')
		.map((s) => s.trim())
		.filter((s) => s !== ''),
)

/** 一段**完全正确**的 MaixPy 程序：校验器不许对它报任何东西。 */
const GOOD = `from maix import camera, display, image, app

cam = camera.Camera(320, 240)
disp = display.Display()

while not app.need_exit():
    img = cam.read()
    blobs = img.find_blobs([[0, 80, 40, 80, 10, 80]], pixels_threshold=500)
    for blob in blobs:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
    disp.show(img)
`

/** 两类最常见的幻觉：编造模块成员、编造实例方法。 */
const HALLUCINATED = `from maix import camera

cam = camera.Camera()
cam.super_zoom()
maix.image.Image.magic_method()
`

/** 带 markdown 围栏的自然语言答案：只有围栏里的算断言。 */
const PROSE = `你可以这样写：

\`\`\`python
from maix import camera
cam = camera.Camera()
\`\`\`

另外 \`maix.no_such_thing\` 只是正文里提到的一个名字，不算断言。
`

const CASES = [
	{
		name: '正确的 MaixPy 程序：一个符号都不许报',
		run: () => checkSymbols(GOOD, roster, { assumeCode: true }),
		expect: (r) => r.checked > 0 && r.unknown.length === 0,
		why: '过报的校验器会把正确代码判成幻觉（教学主线事故 03）。',
	},
	{
		name: '编造的模块成员与实例方法都要抓到',
		run: () => checkSymbols(HALLUCINATED, roster, { assumeCode: true }),
		expect: (r) =>
			r.unknown.includes('maix.image.Image.magic_method') &&
			r.unknown.includes('maix.camera.Camera.super_zoom'),
		why: '`cam.super_zoom()` 只有在理解 `cam = camera.Camera()` 之后才查得出来 —— 单层构造跟踪。',
	},
	{
		name: '枚举成员取值要放行',
		run: () => symbolIsValid('maix.image.Format.FMT_RGB888', roster),
		expect: (r) => r === true,
		why: '白名单通常只登记到枚举类，不放行会把正确代码判成幻觉。',
	},
	{
		name: '类存在、方法不存在 —— 不许因为前缀命中就放行',
		run: () => symbolIsValid('maix.camera.Camera.super_zoom', roster),
		expect: (r) => r === false,
		why: '剥离末段必须只允许「常量形」的段。这是教学主线踩过的假阴性。',
	},
	{
		name: '正文里提到的名字不算断言（assumeCode=false 只看围栏）',
		run: () => checkSymbols(PROSE, roster),
		expect: (r) => !r.refs.includes('maix.no_such_thing') && r.refs.includes('maix.camera.Camera'),
		why: '正文提到一个名字不等于声称它存在。',
	},
	{
		name: '裸代码必须整段都查（assumeCode=true）',
		run: () => checkSymbols('maix.image.Image.magic_method()', roster, { assumeCode: true }),
		expect: (r) => r.checked === 1 && r.unknown.length === 1,
		why: '事故 01：这里返回 0 会被当成「没有可校验的符号」而放行。所以调用方必须显式声明意图。',
	},
	{
		name: '没有 maix 痕迹的文本：checked 为 0',
		run: () => checkSymbols('print("hello")', roster, { assumeCode: true }),
		expect: (r) => r.checked === 0 && r.unknown.length === 0,
		why: '「没什么可查」是一种合法结局，工具据此回答而不是报警。',
	},
	{
		name: 'import maix.image as im 这种别名也要解析',
		run: () => extractSymbolReferences('import maix.image as im\nim.Image()', { assumeCode: true }),
		expect: (r) => r.includes('maix.image.Image'),
		why: 'MaixPy 最常见的写法不出现 `maix.` 前缀，只认全限定名等于对最常见的一类代码失明。',
	},
]

let failed = 0
for (const c of CASES) {
	let actual
	let ok = false
	let note = ''
	try {
		actual = c.run()
		ok = c.expect(actual) === true
	} catch (err) {
		note = `抛异常：${err.message}`
	}
	if (ok) {
		console.log(`  ok   ${c.name}`)
	} else {
		failed += 1
		console.log(`  FAIL ${c.name}`)
		console.log(`       ${c.why}`)
		console.log(`       实际：${note || JSON.stringify(actual)}`)
	}
}

console.log('')
console.log(`白名单 ${roster.size} 条 · ${CASES.length} 个用例 · ${failed} 个失败`)
process.exit(failed === 0 ? 0 : 1)
