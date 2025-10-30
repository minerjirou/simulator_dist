了解です。ここまで一緒に固めてきた**現行戦略のサマリ**を1枚に凝縮します（5機編成・データリンク前提、相手挙動はランダム想定）。

---

# **現状の戦略サマリ（Open部門・VIP戦）**

## **目的と基本方針**

* **勝利条件**：敵護衛対象機（VIP/HVAA）を**先に撃破**、自VIPは**確実に生存**。

* **設計思想**：相手の意図を読むのではなく、**幾何（TTI/LOS）＋ドクトリン準拠**で「守り（CAP）」と「攻め（多軸同時圧力）」を両立。CAPは対空防御の定石で、VIP護衛と相性がよい。 ([U.S. Air Force Doctrine](https://www.doctrine.af.mil/Portals/61/documents/AFDP_3-01/3-01-AFDP-COUNTERAIR.pdf?utm_source=chatgpt.com))

## **編成・役割**

* **Attacker ×2（A要素）**：敵VIPへ\*\*多軸（Wall→Bracket/Pincer）\*\*で接近し、**射後クランク**→F-pole確保→必要ならポンプ→再コミット。 ([ウィキペディア](https://en.wikipedia.org/wiki/Finger-four?utm_source=chatgpt.com))

* **Defender ×2（D要素）**：自VIP周囲に**CAPボックス**（前後スタック）を維持し、**TTI(敵→VIP)**・閉率・アスペクトで**最大脅威**のみ迎撃（**Grinder**でHot/Cold交代）。 ([ウィキペディア](https://en.wikipedia.org/wiki/Combat_air_patrol?utm_source=chatgpt.com))

* **VIP（5機目）**：味方CAPが作る**安全回廊**を維持（敵側に寄り過ぎない）。

## **交戦タイムライン（BVR→WVR）**

1. **初動**：A要素は**Wall**で押し上げ→アジマス拡張し**Bracket**、先着がシューター。

2. **ミサイル射後（ARH化前）**：**クランク**で方位角30–50°保持＝誘導継続しつつ閉距離を抑制。 ([FlyAndWire](https://flyandwire.com/2020/05/06/updated-crank-manoeuvre/?utm_source=chatgpt.com))

3. **ピットブル検知**（ARH能動化）後：**ノッチ/ビーム**（90°±α）＋**短時間ドラッグ**で離隔→**再コミット**。 ([ウィキペディア](https://en.wikipedia.org/wiki/Active_radar_homing?utm_source=chatgpt.com))

4. **近接に落ちた場合**：EM理論に沿って**エネルギー優位**を回復（必要に応じ高/低ヨーヨー）、長居せず再度BVRへ。 ([ウィキペディア](https://en.wikipedia.org/wiki/Energy%E2%80%93maneuverability_theory?utm_source=chatgpt.com))

## **センサー/ターゲティング**

* **データリンク**：全機の最新トラックを**ブラックボード共有**。消失時は**等速外挿（CV）＋信頼度**で疑似トラック維持。

* **選定規則**：

  * Attacker：**敵VIP最優先**（不可なら最近/高脅威の敵戦闘機）。

  * Defender：`Danger = w1/TTI(enemy→VIP)+w2*closing_rate+w3*aspect` 最大を**即迎撃**（ヒステリシスでフラつき防止）。

* **多軸維持**：A1が右にクランクしたらA2は左へ—**常に左右オフセット**で射機会を複数化。 ([codex.uoaf.net](https://codex.uoaf.net/index.php/Air-to-air_basic_tactics?utm_source=chatgpt.com))

## **武器運用**

* **発射判断**：`P_kill(range, aspect, closure)`が閾値超で**サルボ**可／弾薬・時間で調整。

* **射後行動**：**クランク→ポンプ→再コミット**。ARHは**ピットブル**以降は母機支援が不要になるため、自由機動へ移行可能。 ([ウィキペディア](https://en.wikipedia.org/wiki/Active_radar_homing?utm_source=chatgpt.com))

## **防御（MWS/RWRトリガ駆動）**

* **Pre-pitbull**：**クランク優先**（誘導継続・閉距離抑制）。

* **Pitbull**：**ノッチ/ビーム**でドップラー零化を狙い、**短時間ドラッグ**→静穏確認で**再上昇・再コミット**。 ([Reddit](https://www.reddit.com/r/WarCollege/comments/1006a9b/what_are_notchingbeaming_how_useful_are_they_and/?utm_source=chatgpt.com))

## **高度とエネルギー**

* **「高度優位＝必須」ではない**。本質は**EM（高度＋速度の合計）で、状況に応じ高度⇄速度**を交換。

* A要素は**中〜高高度基調**、D要素は**スタックCAP**（中＋やや低）で即応性を担保。 ([ウィキペディア](https://en.wikipedia.org/wiki/Energy%E2%80%93maneuverability_theory?utm_source=chatgpt.com))

## **実装のコア（FSM 4状態）**

1. **HIGH\_ATTACK**：多軸で接近（Wall/Bracket）。

2. **PRE\_PITBULL\_CRANK**：射後〜能動化前は**35°前後クランク**でF-pole延伸。 ([FlyAndWire](https://flyandwire.com/2020/05/06/updated-crank-manoeuvre/?utm_source=chatgpt.com))

3. **PITBULL\_BEAM\_DRAG\_LOW**：90°±θビーム＋短時間ドラッグ。

4. **RECLIMB\_RECOMMIT**：静穏tで再上昇→多軸復帰。

## **評価指標（オフライン検証）**

* **敵VIP撃破率/平均撃破時間**、**自VIP生存率**、**TTI(enemy→VIP)の平均・分布**、**A要素のF-pole分布**、**ミサイル効率**。

* 乱択（初期方位/性能/軌道）でモンテカルロ実験を実施。

---

この設計は、**CAP＋BVRタイムライン（クランク/ノッチ/ポンプ/再コミット）**、**多軸（Bracket/Wall/Grinder）**、**EM理論**といった定石に整合し、**相手がランダムでも**トリガーと幾何量で安定して回る構造です。 ([ウィキペディア](https://en.wikipedia.org/wiki/Combat_air_patrol?utm_source=chatgpt.com))

必要ならこのサマリを基に、あなたの`targeting.py / maneuver.py / evasion.py`へ入れる**最小差分パッチ**をすぐ出します。

