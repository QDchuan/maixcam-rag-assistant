"""真实模型接入：把一次模型调用变成一次「决策」。

## 为什么用 JSON 协议而不是原生 function calling

两种做法都能用，本项目选 JSON 协议，理由有三条：

1. **可移植**：任何 OpenAI 兼容端点都能用，不依赖特定厂商的 function calling 支持；
2. **可见**：模型看到的输出格式就是提示词里那几行，**读者能完整看到**
   agent 是怎么"决定"调工具的，而不是把它藏进 SDK 的参数里；
3. **可测**：协议是纯文本，假模型能精确复现任何一种输出——包括畸形输出。

代价是多了一层解析，而且模型偶尔会不守格式。
所以解析必须**宽容且显式**：能救的救，救不了的说清为什么。

## 一个刻意的容错方向

模型返回的不是 JSON（比如它直接写了一篇散文答案）时，
本实现**把它当作最终答案**，而不是报错。

理由是：这种输出大概率就是答案，把它丢掉再让模型重试，
等于用一个格式问题换掉一次已经正确的回答。**容错要往"少损失"的方向做，
而不是往"严格"的方向做。**
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..corpus.tokenize import count_tokens
from ..providers import ChatClient
from .loop import Decision, Message

# 协议说明。它会拼在系统提示的末尾。
JSON_PROTOCOL = """

────────────────────────────────────────
输出格式（必须严格遵守）

你必须**只输出一个 JSON 对象**，两种形式之一：

调用工具：
{"action": "tool_call", "tool": "<工具名>", "args": {<参数>}}

给出最终答案：
{"action": "final", "answer": "<给用户的回答>"}

规则：
1. 一次只输出一个 JSON 对象，不要有别的文字，不要用代码块包起来。
2. 需要更多信息时用 tool_call；信息够了、或资料里确实没有答案时用 final。
3. `args` 里的参数名必须与工具说明完全一致。
4. 如果已经调用过工具但没查到，**不要反复用同样的参数重试**——
   换个关键词，或者用 final 告诉用户资料中没有。
"""

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_decision(text: str) -> tuple[Decision | None, str]:
    """把模型输出解析成决策。

    返回 (决策, 说明)。决策为 None 表示"没解析出结构，应当当作文本答案处理"。

    宽容策略，按顺序尝试：
      1. 直接 json.loads
      2. 剥掉代码围栏再试
      3. 在文本里找第一个平衡的 `{...}` 再试
      4. 都失败 -> 返回 None，交给调用方当散文答案处理
    """
    raw = (text or "").strip()
    if not raw:
        return None, "模型返回了空内容"

    candidates: list[str] = [raw]

    m = _JSON_BLOCK.search(raw)
    if m:
        candidates.append(m.group(1))

    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start:i + 1])
                    break

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        action = str(data.get("action", "")).strip().lower()

        if action in ("final", "answer"):
            ans = data.get("answer", data.get("text", ""))
            return Decision.final(str(ans)), "解析为最终答案"

        if action in ("tool_call", "tool", "call"):
            name = str(data.get("tool", data.get("name", ""))).strip()
            args = data.get("args", data.get("arguments", {})) or {}
            if not name:
                return None, "tool_call 缺少 tool 字段"
            if not isinstance(args, dict):
                return None, f"args 必须是对象，实际是 {type(args).__name__}"
            return Decision.call(name, args), "解析为工具调用"

        # 有 JSON 但没有 action：可能是模型直接给了 {"answer": ...}
        if "answer" in data:
            return Decision.final(str(data["answer"])), "无 action 字段，按 answer 处理"

    return None, "没有解析出合法的决策结构"


@dataclass
class JsonProtocolModel:
    """通过 OpenAI 兼容端点做决策。

    `temperature=0` 是刻意的：agent 的决策需要可复现，
    随机性会让"同一问题两次跑出不同结果"变成一个无法排查的问题。
    """

    chat: ChatClient
    name: str = "json-protocol"
    # 解析失败的次数，用于观察模型守不守格式
    parse_failures: int = 0
    calls: int = 0

    def decide(self, messages: list[Message],
               tools: list[dict[str, Any]]) -> Decision:
        self.calls += 1

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m for m in messages if m.role != "system"]

        # 把内部消息序列渲染成模型能读的对话。
        # 工具结果明确标出角色，否则模型分不清"这是它的调用"还是"用户说的话"。
        lines: list[str] = []
        for m in turns:
            if m.role == "user":
                lines.append(f"用户：{m.content}")
            elif m.role == "assistant":
                lines.append(f"你：{m.content}")
            elif m.role == "tool":
                lines.append(f"[工具 {m.tool_name or '?'} 的结果]\n{m.content}")
        user_text = "\n\n".join(lines)

        raw = self.chat.complete(system + JSON_PROTOCOL, user_text, json_mode=False)
        tokens = count_tokens(system) + count_tokens(user_text) + count_tokens(raw)

        decision, note = parse_decision(raw)
        if decision is None:
            self.parse_failures += 1
            # 宽容方向：把散文当答案，而不是丢掉一次可能正确的回答
            return Decision.final(
                raw if raw else f"（模型没有给出可用回复：{note}）",
                tokens=tokens,
            )
        return Decision(kind=decision.kind, tool_call=decision.tool_call,
                        text=decision.text, tokens=tokens)

    def stats(self) -> dict[str, Any]:
        rate = (self.parse_failures / self.calls) if self.calls else 0.0
        return {
            "calls": self.calls,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": round(rate, 3),
        }
