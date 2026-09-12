# 分支 B：把 MaixCAM 开发助手做成一个真的 AI 应用

这一目录**不属于**教学主线。教学主线（`maixrag/`、`corpus/`、`indexes/`、`eval/`）回答的是
「检索是怎么做出来的」；这一目录回答的是另一个问题：

> 底座已经成熟了，我们怎么把自己的领域能力做成一个**产品**？

## 一分钟说清楚这一分支的定位

| | 教学主线（分支 A） | 这一分支（分支 B） |
| --- | --- | --- |
| 目标 | 讲清楚原理 | 做出**能用**的助手 |
| Agent 循环 | 自己写（`maixrag/agent/`） | 用 DeepSeek Harness |
| 检索 | 自己实现 BM25 + 向量 + RRF | 接成熟的 |
| 前端 | 终端 REPL | DSH Web 界面 + 领域外壳 |
| 学到 | 每一层为什么这么设计 | **底子是别人的，差异化在你的领域层** |

顺带说清一个曾经理解错、后来改正的地方：这一分支**不是**「给 DSH 写一个 RAG 插件」，
而是「**以 DSH 为基座，做一个 MaixCAM 助手**」。区别在于交付物：前者是一个插件包，
后者是一套装好就能用的助手应用。

## 目录

```
harness/
  dsh-maixcam-shell/     dsh Web 插件包 —— 这就是「壳」
    lib/index.js         宿主半边：一个 Loader 座位
    lib/client.js        浏览器半边：品牌、领域面板（手写的 lazy-CJS bundle，无构建）
    cordis.patch.yml     bundle 层，供 `dsh plugin add` 使用
  tools/eyes.mjs         无头浏览器验证工具：读文本、量元素、截图
  install.py             把壳装进某个 dsh profile（也可 --check / --uninstall）
```

## 装它

```bash
python -m harness.install              # 装到默认 profile（web）
python -m harness.install --check      # 看装没装、装在哪
python -m harness.install --uninstall  # 撤回
```

装完重启 `dsh web`，侧栏就应该从「deepseek / LOCAL BUILD」变成
「MaixCAM 开发助手 / MaixPy」，首屏标题前也是 MaixCAM 的图标。

它**不改 DeepSeek Harness 本身**：只在 `$DSH_HOME/profiles/<profile>/plugins/` 下放一个包，
再往那个 profile 的 `cordis.patch.yml` 里插一段带标记的行。标记之外的字节一个都不碰，
所以别人装在同一个 profile 里的插件不受影响，`--uninstall` 也能干净撤回。

## 「壳」是怎么接上去的

DSH 的 Web 前端有一个**槽位（slot）系统**：布局本身是壳提供的，功能插件往座位
（occupant）里放自己的东西。所以做定制不需要 fork 前端，也不需要重新构建 Web 包。

```
DSH Web 前端（已构建好，不动）
  └─ 槽位 root
       ├─ sidebar
       │    ├─ sidebar.brand.mark    ← 我们占了：MaixCAM 图标
       │    └─ sidebar.brand.name    ← 我们占了：MaixCAM 开发助手
       ├─ main.conversation
       │    ├─ conversation.hero.brand.mark  ← 我们占了：首屏图标
       │    ├─ conversation.composer.dock    ← 待占：领域入口
       │    └─ ...
       └─ ...
```

插件包只要在 `package.json` 里声明 `dsh.client`，并把浏览器半边放在 `exports["./client"]`，
宿主就会把它当浏览器插件加载 —— **没有构建步骤**。浏览器半边直接用宿主自己的
lazy-CJS bundle 格式手写：

```js
window.__ModuleLoader__.load({
  id: 'dsh-maixcam-shell',
  factory: (require) => {
    const module = { exports: {} }
    const React = require('react')
    // ...组件与 ctx.slots.register...
    exports.apply = apply
    exports.inject = ['slots']
    return module.exports
  },
})
```

## 两个踩过的坑（都是静默失效）

### 1. 「填座位」和「换座位」不是一回事

`single` 座位的选举规则是：**按优先级排序后，第一个存活的条目获胜** ——
也就是**数字越小越优先**。

而壳的默认品牌（侧栏的鱼形标志与「LOCAL BUILD」标签）**不是兜底渲染，而是一个
`priority: 0` 的在册占位者**。所以：

- `conversation.hero.brand.mark` 原本没人占 → 登记就赢；
- `sidebar.brand.mark` 已有占位者 → 并列 0 时先登记的默认占位者赢，
  我们的组件**根本不渲染，而且不报错**，页面看起来「什么都没发生」。

正确做法是显式压过它：

```js
ctx.slots.register({ name: 'sidebar.brand.mark', priority: -1 }, MaixCamMark)
```

实测确认：`priority: -1` 赢，`priority: +1` 输。

> **教训**：界面扩展最难的从来不是「怎么画」，而是「我画的东西到底有没有被渲染」。
> 一个不报错的失败，比一个报错的失败贵得多。

### 2. 验证前端的反馈必须自己拿得到

前端的反馈如果是「叫用户刷新、肉眼看、再口述回来」，一轮要几分钟，而且看不清
「是哪个座位没占上」。所以这个分支带了一个工具：

```bash
node harness/tools/eyes.mjs --url <URL> --text --shot out.png
node harness/tools/eyes.mjs --url <URL> --eval "document.querySelector('.dsh-mx-name')?.innerText"
```

它起一个无头 Chrome，用 DevTools 协议连上，把「文本 / 元素几何 / 截图」拿回来。
上面那个优先级坑，就是它一次实验定位的，没有麻烦任何人。

配合 `dsh web --port 0 --no-open`（打印一个带 token 的 URL）就能起一个**只给自己看**的
验证实例，和用户正在用的那个互不打扰。

## 还没做的

- RAG 接入点（检索服务 / 工具 / 证据面板）
- 领域面板：检索轨迹、语料地图
- `maixcam` settings 命名空间
- Agent preset：MaixCAM 助手的回答纪律

见 [`../docs/design/06-项目结构与两条分支.md`](../docs/design/06-项目结构与两条分支.md) §3。
