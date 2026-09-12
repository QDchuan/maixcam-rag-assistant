/**
 * `dsh-maixcam-shell` — 宿主半边。
 *
 * 这个包交付的是**界面**，不是 agent 能力。DeepSeek Harness 的 agent 循环、
 * 工具系统、会话管理都已经成熟，不需要重写；要换掉的是这层通用外壳 ——
 * 让用户看到的是「MaixCAM 开发助手」，而不是「一个通用 coding agent」。
 *
 * 浏览器半边（`./client.js`）通过 `exports["./client"]` 交付，并且是**手写的
 * lazy-CJS bundle**，所以这个包没有任何构建步骤：宿主把这个文件原样当
 * `./client` 导出发给浏览器。
 *
 * 宿主半边目前只是一个 Loader 座位：界面插件在宿主侧不需要自己的服务或路由。
 * 领域检索的接入点在后续版本里落到这里（`dsh.client` 声明的界面与本文件的
 * 路由共享一份配置）。
 */

/**
 * 宿主插件体。
 *
 * 空实现是有意的：一个 `@deepseek-ai/dsh-client-ui-*` 风格的界面包在宿主侧
 * 只需要占住 Loader 的一行，浏览器半边才是有内容的那一半。
 *
 * @param ctx - 宿主插件上下文。
 */
function apply(ctx) {
  ctx.logger?.info?.('dsh-maixcam-shell: 外壳已挂载（MaixCAM 开发助手）')
}

export { apply }
