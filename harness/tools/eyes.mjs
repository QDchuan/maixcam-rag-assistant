/**
 * 给「壳」装一双眼睛：无头浏览器打开一个页面，读回文本、量元素、截图。
 *
 * ## 为什么需要它
 *
 * 前端的反馈必须是自己能拿到的。靠人肉刷新页面、肉眼看、再口述回来，
 * 一轮要几分钟，而且看不清「哪个座位没占上」「哪一屏读起来别扭」。
 * 这个脚本把那一轮压到几秒，并且给出**可比对的结构化结果**——
 * 文本是不是出现了、元素在不在、宽高是多少。
 *
 * 它做的事很小：起一个无头 Chrome，用 DevTools 协议连上，求值一段表达式，
 * 顺手截一张图，然后关掉。没有框架、没有依赖（Node 22 自带 fetch 与 WebSocket）。
 *
 * ## 用法
 *
 * ```bash
 * # 看品牌行渲染成了什么，并留一张图
 * node harness/tools/eyes.mjs --url "$URL" --text --shot out/brand.png
 *
 * # 量一个元素的几何
 * node harness/tools/eyes.mjs --url "$URL" \
 *   --eval "JSON.stringify(document.querySelector('.dsh-mx-name')?.getBoundingClientRect())"
 * ```
 *
 * `--text` 读整页可见文本；`--eval` 求值任意表达式；两个都给就先求值再读文本。
 * `--wait-for` 给一个表达式，轮询到它为真为止（默认等 `body` 有内容），
 * 这样就不必猜「到底要等几秒」。
 */

import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

/** 常见 Chrome / Edge 安装位置（Windows 优先，其次 POSIX）。 */
const BROWSER_CANDIDATES = [
	process.env.CHROME_PATH,
	'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
	'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
	'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
	'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
	'/usr/bin/google-chrome',
	'/usr/bin/chromium',
	'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean)

/**
 * 解析命令行参数成一张表。
 *
 * @param argv - `process.argv.slice(2)`。
 * @returns 选项对象。
 */
function parseArgs(argv) {
	const opts = {
		url: '',
		eval: '',
		text: false,
		shot: '',
		waitFor: '',
		width: 1440,
		height: 900,
		timeoutMs: 30000,
		settleMs: 900,
		keep: false,
	}
	for (let i = 0; i < argv.length; i += 1) {
		const a = argv[i]
		const next = () => argv[(i += 1)]
		if (a === '--url') opts.url = next()
		else if (a === '--eval') opts.eval = next()
		else if (a === '--shot') opts.shot = next()
		else if (a === '--wait-for') opts.waitFor = next()
		else if (a === '--width') opts.width = Number(next())
		else if (a === '--height') opts.height = Number(next())
		else if (a === '--timeout') opts.timeoutMs = Number(next())
		else if (a === '--settle') opts.settleMs = Number(next())
		else if (a === '--text') opts.text = true
		else if (a === '--keep') opts.keep = true
		else if (a === '--help' || a === '-h') opts.help = true
		else throw new Error(`不认识的参数：${a}`)
	}
	return opts
}

/**
 * 找一个本机可用的高位端口。
 *
 * @returns 端口号。
 */
function freePort() {
	return 9200 + Math.floor(Math.random() * 600)
}

/**
 * 轮询 HTTP 端点直到拿到 JSON。
 *
 * @param url - 目标地址。
 * @param attempts - 最多试几次。
 * @returns 解析出的 JSON。
 */
async function pollJson(url, attempts = 100) {
	let last = null
	for (let i = 0; i < attempts; i += 1) {
		try {
			const res = await fetch(url)
			if (res.ok) return await res.json()
			last = new Error(`HTTP ${res.status}`)
		} catch (err) {
			last = err
		}
		await sleep(100)
	}
	throw new Error(`连不上 ${url}：${last?.message ?? '未知错误'}`)
}

/**
 * 睡一会儿。
 *
 * @param ms - 毫秒。
 * @returns 睡眠完成的 promise。
 */
function sleep(ms) {
	return new Promise((r) => setTimeout(r, ms))
}

/**
 * 一个极简的 DevTools 协议客户端：发一条命令，等它的回包。
 */
class Cdp {
	/**
	 * @param ws - 已经连上的 WebSocket。
	 */
	constructor(ws) {
		this.ws = ws
		this.seq = 0
		this.pending = new Map()
		ws.addEventListener('message', (event) => {
			const msg = JSON.parse(event.data)
			if (msg.id === undefined) return
			const slot = this.pending.get(msg.id)
			if (slot === undefined) return
			this.pending.delete(msg.id)
			if (msg.error) slot.reject(new Error(`${msg.error.message} (${msg.error.code})`))
			else slot.resolve(msg.result)
		})
	}

	/**
	 * 发一条命令。
	 *
	 * @param method - CDP 方法名。
	 * @param params - 方法参数。
	 * @returns 结果。
	 */
	send(method, params = {}) {
		this.seq += 1
		const id = this.seq
		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve, reject })
			this.ws.send(JSON.stringify({ id, method, params }))
			setTimeout(() => {
				if (this.pending.delete(id)) reject(new Error(`${method} 超时`))
			}, 30000)
		})
	}

	/**
	 * 在页面里求值一个表达式并取回它的值。
	 *
	 * @param expression - 要求值的 JS 源码。
	 * @returns 求值结果（已按值取回）。
	 */
	async eval(expression) {
		const res = await this.send('Runtime.evaluate', {
			expression,
			returnByValue: true,
			awaitPromise: true,
		})
		if (res.exceptionDetails) {
			const text = res.exceptionDetails.exception?.description ?? res.exceptionDetails.text
			throw new Error(`页面里求值失败：${text}`)
		}
		return res.result?.value
	}
}

/**
 * 主流程：起浏览器、连上、求值、截图、收工。
 *
 * @param opts - 命令行选项。
 * @returns 进程退出码。
 */
async function main(opts) {
	if (opts.help || !opts.url) {
		console.log(
			[
				'用法：node harness/tools/eyes.mjs --url <URL> [选项]',
				'',
				'  --text            读回整页可见文本',
				'  --eval <expr>     在页面里求值一个表达式',
				'  --wait-for <expr> 轮询到该表达式为真（默认等 body 有内容）',
				'  --shot <path>     截图写到这里',
				'  --width/--height  视口尺寸（默认 1440x900）',
				'  --timeout <ms>    等待上限（默认 30000）',
				'  --settle <ms>     满足等待条件后再停多久（默认 900，给字体和动画）',
				'  --keep            收起浏览器后不删用户目录（调试用）',
			].join('\n'),
		)
		return opts.url ? 0 : 2
	}

	const port = freePort()
	const userDataDir = resolve(
		process.env.TEMP ?? '/tmp',
		`dsh-eyes-${process.pid}-${port}`,
	)
	mkdirSync(userDataDir, { recursive: true })

	const bin = BROWSER_CANDIDATES.find((p) => existsSync(p))
	if (bin === undefined) throw new Error('找不到 Chrome / Edge；用 CHROME_PATH 指定一个')

	const child = spawn(
		bin,
		[
			'--headless',
			'--disable-gpu',
			'--no-sandbox',
			'--no-first-run',
			'--no-default-browser-check',
			'--disable-extensions',
			'--disable-background-networking',
			'--hide-scrollbars',
			`--remote-debugging-port=${port}`,
			`--user-data-dir=${userDataDir}`,
			`--window-size=${opts.width},${opts.height}`,
			opts.url,
		],
		{ stdio: 'ignore' },
	)

	let ws
	try {
		const list = await pollJson(`http://127.0.0.1:${port}/json/list`)
		const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
		if (page === undefined) throw new Error('浏览器没有给出可连的页面目标')

		ws = new WebSocket(page.webSocketDebuggerUrl)
		await new Promise((res, rej) => {
			ws.addEventListener('open', res, { once: true })
			ws.addEventListener('error', () => rej(new Error('连 DevTools 失败')), { once: true })
		})

		const cdp = new Cdp(ws)
		await cdp.send('Runtime.enable')
		await cdp.send('Page.enable')

		const condition =
			opts.waitFor || "document.body && document.body.innerText.trim().length > 40"
		const deadline = Date.now() + opts.timeoutMs
		let ready = false
		while (Date.now() < deadline) {
			try {
				if ((await cdp.eval(`!!(${condition})`)) === true) {
					ready = true
					break
				}
			} catch {
				// 页面还在导航，求值会失败；继续等。
			}
			await sleep(150)
		}
		await sleep(opts.settleMs)

		console.log(`ready=${ready}  url=${opts.url}`)

		if (opts.eval) {
			const value = await cdp.eval(opts.eval)
			console.log('--- eval ---')
			console.log(typeof value === 'string' ? value : JSON.stringify(value, null, 2))
		}

		if (opts.text) {
			const text = await cdp.eval('document.body.innerText')
			console.log('--- text ---')
			console.log(text)
		}

		if (opts.shot) {
			const res = await cdp.send('Page.captureScreenshot', { format: 'png' })
			const target = resolve(opts.shot)
			mkdirSync(dirname(target), { recursive: true })
			writeFileSync(target, Buffer.from(res.data, 'base64'))
			console.log(`shot -> ${target}`)
		}

		return ready ? 0 : 1
	} finally {
		try {
			ws?.close()
		} catch {
			// 关不掉就算了，下面直接把浏览器收掉。
		}
		if (!opts.keep) child.kill()
	}
}

// 直接跑时走 main；被 import 时只导出工具函数。
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main(parseArgs(process.argv.slice(2)))
		.then((code) => process.exit(code))
		.catch((err) => {
			console.error(`eyes: ${err.message}`)
			process.exit(1)
		})
}
