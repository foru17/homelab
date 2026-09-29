# 分流：按需求选出口

一台机器解决不了所有需求。按服务把流量送到合适地区的出口，规则放在香港 hub（或家里网关）统一维护。

## 常见需求 → 出口

| 需求 | 建议出口 | 备注 |
|---|---|---|
| 游戏、低延迟、日常网页 | 香港（机房 IP 或家宽） | 离得近，延迟最低 |
| GitHub / Docker / 依赖拉取 | 香港机房直出 | 看重单连接带宽 |
| YouTube | 香港 / 任意 | 带宽优先 |
| Netflix | 台湾 / 新加坡 / 美国（按片库需要） | 机房 IP 常被识别，家宽或原生 IP 更稳 |
| Spotify | 日本 / 美国等目标区 | 账号地区与出口一致 |
| TikTok | 日本 / 新加坡 / 美国 | 香港不可用 |
| ChatGPT / Claude / Gemini 等 AI | 美国（机房 + WARP 兜底）或日本 | 固定一个地区，别频繁切换 |
| 对机房 IP 不友好的服务 | 各地家宽出口 | 只出不进 |
| 国内服务 | 直连 | 别让国内流量绕出去 |

## mihomo 分组骨架（占位名，按需改）

```yaml
proxy-groups:
  - {name: 跨境专线, type: url-test, proxies: [<LINE_A>, <LINE_B>, <LINE_C>], url: http://www.gstatic.com/generate_204, interval: 300, tolerance: 20}
  - {name: 香港出口, type: select, proxies: [跨境专线, <HK_RESIDENTIAL>, <HK_WARP>]}
  - {name: 日本出口, type: select, proxies: [<JP_EXIT>]}
  - {name: 新加坡出口, type: select, proxies: [<SG_EXIT>]}
  - {name: 美国出口, type: url-test, proxies: [<US_CN2GIA>, <US_WARP>], url: http://www.gstatic.com/generate_204, interval: 300}
  - {name: AI, type: select, proxies: [美国出口, 日本出口]}
  - {name: 流媒体, type: select, proxies: [新加坡出口, 日本出口, 美国出口, 香港出口]}
rules:
  - GEOSITE,openai,AI
  - GEOSITE,anthropic,AI
  - GEOSITE,netflix,流媒体
  - GEOSITE,spotify,日本出口
  - GEOSITE,tiktok,日本出口
  - GEOSITE,youtube,香港出口
  - GEOSITE,cn,DIRECT
  - GEOIP,CN,DIRECT
  - MATCH,香港出口
```

要点：

- 组之间不要互相引用成环（有的客户端会直接报错）。
- 别人提供、不归你独占的出口单独成组，**不要**放进「自动选择」一类的组，避免别人的流量误走。
- 规则集用远程 rule-provider，客户端本地缓存；别把大名单打包进配置。
