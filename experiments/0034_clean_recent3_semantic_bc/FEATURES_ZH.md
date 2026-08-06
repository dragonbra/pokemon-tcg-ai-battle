# 0032 鏈€缁堢壒寰佽〃

鐘舵€佽鏄庯細鏈〃涓殑鈥滀繚鐣欌€濆瓧娈靛叏閮ㄧ粡杩?`compiler -> batch -> forward` 鏍￠獙锛涘垹闄ら」涓嶄細杩涘叆 actor batch銆?
## 鍏ㄥ眬鍐崇瓥鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `select_type` | 閫夋嫨绫诲瀷 | 褰撳墠寮曟搸瑕佹眰鐜╁瀹屾垚鐨勯€夋嫨绫诲埆 |
| `select_context` | 閫夋嫨涓婁笅鏂?| 褰撳墠閫夋嫨鍙戠敓鍦ㄥ摢绉嶈鍒欎笂涓嬫枃 |
| `relative_first_player` | 鐩稿鍏堟墜鏂?| 鍏堟墜鏄嚜宸辫繕鏄鎵?|
| `supporter_played` | 鏈洖鍚堝凡鐢ㄦ敮鎻磋€?| 鏈洖鍚堟槸鍚﹀凡缁忎娇鐢ㄦ敮鎻磋€呭崱 |
| `stadium_played` | 鏈洖鍚堝凡鍑虹珵鎶€鍦?| 鏈洖鍚堟槸鍚﹀凡缁忔墦鍑虹珵鎶€鍦?|
| `energy_attached` | 鏈洖鍚堝凡鎵嬭创鑳介噺 | 鏈洖鍚堥€氬父鑳介噺璐撮檮鏈轰細鏄惁宸蹭娇鐢?|
| `retreated` | 鏈洖鍚堝凡鎾ら€€ | 鏈洖鍚堟挙閫€鏈轰細鏄惁宸蹭娇鐢?|
| `turn` | 鍥炲悎缂栧彿 | 褰撳墠鎬诲洖鍚堢紪鍙?|
| `turn_action_count` | 鍥炲悎鍐呮搷浣滄暟 | 褰撳墠鍥炲悎宸叉墽琛岀殑鎿嶄綔鏁伴噺 |
| `own_deck_count` | 宸辨柟鐗屽簱寮犳暟 | 宸辨柟鐗屽簱鍓╀綑鍗℃暟 |
| `opponent_deck_count` | 瀵规墜鐗屽簱寮犳暟 | 瀵规墜鐗屽簱鍓╀綑鍗℃暟 |
| `opponent_hand_count` | 瀵规墜鎵嬬墝鏁?| 瀵规墜褰撳墠鎵嬬墝鏁伴噺 |
| `own_prize_count` | 宸辨柟濂栬祻鍗℃暟 | 宸辨柟鍓╀綑濂栬祻鍗℃暟閲?|
| `opponent_prize_count` | 瀵规墜濂栬祻鍗℃暟 | 瀵规墜鍓╀綑濂栬祻鍗℃暟閲?|
| `own_bench_max` | 宸辨柟鍚庡満涓婇檺 | 褰撳墠瑙勫垯鏁堟灉涓嬪繁鏂瑰悗鍦哄閲?|
| `opponent_bench_max` | 瀵规墜鍚庡満涓婇檺 | 褰撳墠瑙勫垯鏁堟灉涓嬪鎵嬪悗鍦哄閲?|
| `remaining_damage_counter` | 鍓╀綑浼ゅ鎸囩ず鐗?| 褰撳墠澶氭閫夋嫨灏氶渶鍒嗛厤鐨勪激瀹虫寚绀虹墿鏁?|
| `remaining_energy_cost` | 鍓╀綑鑳介噺璐圭敤 | 褰撳墠澶氭閫夋嫨灏氶渶澶勭悊鐨勮兘閲忚垂鐢?|

## 鍗＄墝瀹炰緥鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `card_id` | 鍗＄墝鍘熷瀷缂栧彿 | 鐢ㄤ簬鍞竴鏌ユ壘鍗＄墝 prototype锛涗笉鍐嶉澶栧仛绗簩濂楄韩浠?embedding |
| `relative_owner` | 鐩稿鎸佹湁鑰?| 璇ュ疄浣撳睘浜庤嚜宸便€佸鎵嬫垨鏈煡鏂?|
| `zone` | 鎵€鍦ㄥ尯鍩?| owner-neutral 鐨勫墠鍦恒€佸悗鍦恒€佹墜鐗屻€佸純鐗岀瓑鍖哄煙 |
| `zone_slot` | 鍖哄煙妲戒綅 | 鍚屼竴鍖哄煙鍐呯殑鍑嗙‘浣嶇疆 |
| `status_bits` | 鐗规畩鐘舵€佷綅 | 鐫＄湢銆佺伡浼ゃ€佹贩涔便€侀夯鐥广€佷腑姣掞紱鏄庣‘鏃犵姸鎬佷笌涓嶉€傜敤涓嶅悓 |
| `appeared_this_turn_state` | 鏈洖鍚堢櫥鍦虹姸鎬?| 鏈櫥鍦恒€佸凡鐧诲満銆佹湭鐭ユ垨涓嶉€傜敤 |
| `current_hp` | 褰撳墠 HP | observation 涓殑褰撳墠 HP |
| `maximum_hp` | 褰撳墠鏈€澶?HP | 璁″叆宸茬敓鏁堣鍒欏悗鐨?observation 鏈€澶?HP |
| `resolved_energy_type_0_count` | 鏈夋晥鏃犺壊鑳介噺浠芥暟 | 寮曟搸宸茶В鏋愪负 Colorless 鐨勪唤鏁?|
| `resolved_energy_type_1_count` | 鏈夋晥鑽夎兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Grass 鐨勪唤鏁?|
| `resolved_energy_type_2_count` | 鏈夋晥鐏兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Fire 鐨勪唤鏁?|
| `resolved_energy_type_3_count` | 鏈夋晥姘磋兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Water 鐨勪唤鏁?|
| `resolved_energy_type_4_count` | 鏈夋晥闆疯兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Lightning 鐨勪唤鏁?|
| `resolved_energy_type_5_count` | 鏈夋晥瓒呰兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Psychic 鐨勪唤鏁?|
| `resolved_energy_type_6_count` | 鏈夋晥鏂楄兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Fighting 鐨勪唤鏁?|
| `resolved_energy_type_7_count` | 鏈夋晥鎭惰兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Darkness 鐨勪唤鏁?|
| `resolved_energy_type_8_count` | 鏈夋晥閽㈣兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Metal 鐨勪唤鏁?|
| `resolved_energy_type_9_count` | 鏈夋晥榫欒兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Dragon 鐨勪唤鏁?|
| `resolved_energy_type_10_count` | 鏈夋晥鍏ㄥ睘鎬ц兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 All 鐨勪唤鏁?|
| `resolved_energy_type_11_count` | 鏈夋晥瓒?鎭跺鍚堣兘閲忎唤鏁?| 寮曟搸宸茶В鏋愪负 Psychic or Darkness 鐨勫鍚堜唤鏁?|

## 宸辨柟璧勬簮璐︽湰鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `card_id` | 璧勬簮鍗＄墝缂栧彿 | 褰撳墠璐︽湰鏉＄洰瀵瑰簲鐨勫崱鐗?prototype |
| `deck_knowledge` | 鐗屽簱鐭ヨ瘑鐘舵€?| 褰撳墠鐗屽簱鏁伴噺鏄彲瑙併€佽蹇嗐€佺簿纭帹鏂€佹湁鐣岃繕鏄湭鐭?|
| `prize_knowledge` | 濂栬祻鍖虹煡璇嗙姸鎬?| 褰撳墠濂栬祻鍖烘暟閲忕殑鐭ヨ瘑鐘舵€?|
| `initial_count` | 娉ㄥ唽鍗＄粍鍒濆寮犳暟 | 60 鍗℃敞鍐屽崱缁勪腑璇ュ崱鐨勫垵濮嬫暟閲?|
| `visible_playing` | 缁撶畻涓彲瑙佹暟 | 璇ュ崱褰撳墠澶勪簬瑙勫垯缁撶畻涓婁笅鏂囩殑鏁伴噺 |
| `deck_value` | 鐗屽簱绮剧‘鏁?| 绮剧‘宸茬煡鏃惰鍗″湪鐗屽簱鐨勬暟閲?|
| `deck_lower` | 鐗屽簱鏁伴噺涓嬬晫 | 涓嶅畬鍏ㄤ俊鎭笅璇ュ崱鍦ㄧ墝搴撶殑鏈€灏忓彲鑳芥暟閲?|
| `deck_upper` | 鐗屽簱鏁伴噺涓婄晫 | 涓嶅畬鍏ㄤ俊鎭笅璇ュ崱鍦ㄧ墝搴撶殑鏈€澶у彲鑳芥暟閲?|
| `prize_value` | 濂栬祻鍖虹簿纭暟 | 绮剧‘宸茬煡鏃惰鍗″湪濂栬祻鍖虹殑鏁伴噺 |
| `prize_lower` | 濂栬祻鍖烘暟閲忎笅鐣?| 璇ュ崱鍦ㄥ璧忓尯鐨勬渶灏忓彲鑳芥暟閲?|
| `prize_upper` | 濂栬祻鍖烘暟閲忎笂鐣?| 璇ュ崱鍦ㄥ璧忓尯鐨勬渶澶у彲鑳芥暟閲?|
| `deck_information_age` | 鐗屽簱淇℃伅骞撮緞 | 璇ョ墝搴撲俊鎭窛褰撳墠鍐崇瓥鐨勪簨浠跺勾榫?|
| `prize_information_age` | 濂栬祻鍖轰俊鎭勾榫?| 璇ュ璧忓尯淇℃伅璺濆綋鍓嶅喅绛栫殑浜嬩欢骞撮緞 |

## 鍥犳灉浜嬩欢鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `log_type` | 鏃ュ織浜嬩欢绫诲瀷 | 鎽哥墝銆佺Щ鍔ㄣ€佽创闄勩€佹敾鍑汇€丠P 鍙樺寲绛変簨浠剁被鍒?|
| `relative_actor` | 鐩稿鎵ц鑰?| 浜嬩欢鐢辫嚜宸便€佸鎵嬫垨鏈煡鏂规墽琛?|
| `card_id` | 浜嬩欢鏉ユ簮鍗?| 浜嬩欢鐩存帴鍏宠仈鐨勬潵婧愬崱鐗?prototype |
| `target_card_id` | 浜嬩欢鐩爣鍗?| 浜嬩欢鐩存帴鍏宠仈鐨勭洰鏍囧崱鐗?prototype |
| `attack_id` | 浜嬩欢鏀诲嚮缂栧彿 | 鏀诲嚮浜嬩欢鎵€浣跨敤鐨?attack prototype |
| `from_area` | 鏉ユ簮鍖哄煙 | 鍗＄墝绉诲姩鍓嶆墍鍦ㄧ殑寮曟搸鍖哄煙 |
| `to_area` | 鐩爣鍖哄煙 | 鍗＄墝绉诲姩鍚庢墍鍦ㄧ殑寮曟搸鍖哄煙 |
| `active_card_id` | 鍓嶅満鍗＄紪鍙?| 鍒囨崲浜嬩欢涓殑鍓嶅満鍗?prototype |
| `bench_card_id` | 鍚庡満鍗＄紪鍙?| 鍒囨崲浜嬩欢涓殑鍚庡満鍗?prototype |
| `before_card_id` | 鍙樺寲鍓嶅崱缂栧彿 | 杩涘寲鎴栧彉鍖栧墠鐨勫崱鐗?prototype |
| `after_card_id` | 鍙樺寲鍚庡崱缂栧彿 | 杩涘寲鎴栧彉鍖栧悗鐨勫崱鐗?prototype |
| `is_recover` | 鏄惁鍥炲 | HP 鍙樺寲鏄惁涓哄洖澶?|
| `put_damage_counter` | 鏄惁鏀剧疆浼ゅ鎸囩ず鐗?| 瀹樻柟鏃ュ織鐨?`putDamageCounter` 甯冨皵鏍囧織锛涙湭鐭ャ€佸惁銆佹槸涓夋€佸垎寮€缂栫爜 |
| `coin_head` | 鎶曞竵鏄惁姝ｉ潰 | 瀹樻柟鏃ュ織鍗曟鎶曞竵鐨?`head` 甯冨皵缁撴灉锛涙湭鐭ャ€佸弽闈€佹闈笁鎬佸垎寮€缂栫爜 |
| `special_condition_type` | 鐗规畩鐘舵€佺被鍨?| 浜嬩欢鏂藉姞鎴栧鐞嗙殑鐗规畩鐘舵€?|
| `result_type` | 缁撴灉绫诲瀷 | 寮曟搸鍏紑鏃ュ織涓殑绂绘暎缁撴灉缂栧彿 |
| `reason_type` | 鍘熷洜绫诲瀷 | 寮曟搸鍏紑鏃ュ織涓殑绂绘暎鍘熷洜缂栧彿 |
| `age` | 浜嬩欢骞撮緞 | 璇ヤ簨浠惰窛鏈€鏂板彲瑙佷簨浠剁殑璺濈 |
| `value` | 浜嬩欢鏁板€?| 鏃ュ織鍏紑鐨勯€氱敤鏁板€?|
| `count` | 浜嬩欢璁℃暟 | 鏃ュ織鍏紑鐨?count 鍙傛暟 |
| `number` | 浜嬩欢缂栧彿鍊?| 鏃ュ織鍏紑鐨?number 鍙傛暟 |

## 鍚堟硶閫夐」鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `action_type` | 鎿嶄綔绫诲瀷 | 褰撳墠鍚堟硶閫夐」瀵瑰簲鐨勫紩鎿庢搷浣滅被鍒?|
| `source_owner` | 鏉ユ簮鎸佹湁鑰?| 鎿嶄綔鏉ユ簮灞炰簬鑷繁杩樻槸瀵规墜 |
| `source_area` | 鏉ユ簮鍖哄煙 | 鎿嶄綔鏉ユ簮鎵€鍦ㄥ紩鎿庡尯鍩?|
| `target_owner` | 鐩爣鎸佹湁鑰?| 鎿嶄綔鐩爣灞炰簬鑷繁杩樻槸瀵规墜 |
| `target_area` | 鐩爣鍖哄煙 | 鎿嶄綔鐩爣鎵€鍦ㄥ紩鎿庡尯鍩?|
| `source_card_id` | 鏉ユ簮鍗＄墝缂栧彿 | 鏉ユ簮鍗＄墝 prototype锛涘叧绯荤己澶辨椂浠嶅彲淇濈暀璇箟 |
| `target_card_id` | 鐩爣鍗＄墝缂栧彿 | 鐩爣鍗＄墝 prototype |
| `attack_id` | 鏀诲嚮缂栧彿 | 璇ラ€夐」瀵瑰簲鐨?attack prototype |
| `special_condition_type` | 鐗规畩鐘舵€侀€夐」 | 璇ラ€夐」閫夋嫨鐨勭壒娈婄姸鎬佺被鍨?|
| `context_card_id` | 涓婁笅鏂囧崱鐗岀紪鍙?| 鍙戣捣褰撳墠閫夋嫨鐨勪笂涓嬫枃鍗＄墝 prototype |
| `effect_card_id` | 鐢熸晥鍗＄墝缂栧彿 | 褰撳墠缁撶畻鍏宠仈鐨勫崱鐗?prototype锛屼笉鏄?effect node |
| `number` | 閫夐」鏁板€?| 寮曟搸閫夐」鍏紑鐨?number 鍙傛暟 |
| `count` | 閫夐」璁℃暟 | 寮曟搸閫夐」鍏紑鐨?count 鍙傛暟 |

## 鍏崇郴鐗瑰緛

| 瀛楁 | 涓枃鍚嶇О | 鍚箟 |
|---|---|---|
| `card_parent` | 鍗＄墝闄勫睘鍏崇郴 | Energy銆乀ool銆佽繘鍖栧崱鍏蜂綋闄勫睘浜庡摢鍙疂鍙ⅵ |
| `event_source` | 浜嬩欢鏉ユ簮瀹炰緥 | 浜嬩欢鏉ユ簮缁戝畾鍒板綋鍓?observation 涓殑鍏蜂綋鍗″疄渚?|
| `event_target` | 浜嬩欢鐩爣瀹炰緥 | 浜嬩欢鐩爣缁戝畾鍒板叿浣撳崱瀹炰緥 |
| `option_source` | 閫夐」鏉ユ簮瀹炰緥 | Attach 绛夐€夐」鏉ヨ嚜鍝紶鍏蜂綋鑳介噺鍗?|
| `option_target` | 閫夐」鐩爣瀹炰緥 | Attach 绛夐€夐」瑕佷綔鐢ㄤ簬鍝彧鍏蜂綋瀹濆彲姊?|

## Prototype 鐗瑰緛

- 鍗＄墝锛氬崱绉嶃€佸疂鍙ⅵ灞炴€с€佽繘鍖栭樁娈点€佽繘鍖栨潵婧愩€佽鍒?娴佹淳鏍囧織銆? 浣嶄緵鑳藉睘鎬с€? 浣嶅急鐐广€? 浣嶆姉鎬с€佸熀纭€ HP銆佹挙閫€璐圭敤銆佸紩鎿庤兘閲忎唤鏁般€佸崱鐗屽埌鎶€鑳姐€佸崱鐗屽埌鏀诲嚮銆?- 鏀诲嚮锛?0 浣嶆敾鍑昏鍒欐爣蹇椼€? 涓湁搴忛渶姹傝兘閲忔Ы銆佸彇娑堝け璐ヨ涓恒€佸熀纭€浼ゅ锛涙墍灞炲崱鐗岀敱鍗＄墝鍒版敾鍑荤殑鍗曞悜鍏崇郴鎭㈠銆?- 鎶€鑳斤細鎶€鑳界被鍒€?5 浣嶇敓鏁堝尯鍩熴€? 涓妧鑳芥爣蹇椼€佹渶澶氫袱涓Е鍙戝櫒锛涙墍灞炲崱鐗岀敱鍗＄墝鍒版妧鑳界殑鍗曞悜鍏崇郴鎭㈠銆?- 姣忎釜瑙﹀彂鍣細瑙﹀彂绫诲瀷銆佺洰鏍囩帺瀹躲€乣not_me`銆乣skip_enemy_target`銆佷袱涓湁搴忓尯鍩熴€佷袱涓湁搴忕瓫閫夋潯浠讹紱姣忎釜鏉′欢鍚被鍨嬨€佹瘮杈冪銆佸悕绉般€乿alue銆乿alue2銆?- 寮曟搸闈欐€?`Trigger` 鍙湁 `trigger_type` 鍜屼竴浠藉畬鏁寸殑 `subject` 鐩爣瑙勫垯锛屾病鏈夌嫭绔嬮潤鎬?`trigger_target`銆傝繍琛屾椂鐪熸鍛戒腑鐨?`TriggerInfo.subject/object` 涓嶅湪瀹樻柟 actor observation 涓紝涓嶈兘浼€狅紱鍏紑鏃ュ織涓彲瑙佺殑鍏蜂綋瀹炰緥鐢?`event_source/event_target` 琛ㄨ揪銆?- 鍗＄墝瑙勫垯/娴佹淳鏍囧織閫愰」涓猴細`can_play_first_turn` 棣栧洖鍚堝彲鐢ㄣ€乣can_trash` 鍙富鍔ㄥ純缃€乣transform_only` 浠呭彉韬娇鐢ㄣ€乣trash_my_turn_end` 鍥炲悎鏈純缃€乣cannot_to_hand_or_deck_in_trash` 寮冪墝鍖轰笉鍙洖鏀躲€乣tera` 澶櫠銆乣to_bench` 鐩存帴鍘诲悗鍦恒€乣to_battle_field_only_setup` 寮€灞€浠呭墠鍦恒€乣to_active_only_setup` 寮€灞€浠呮垬鏂楀満銆乣no_prize` 涓嶅彇濂栬祻銆乣only_team_rocket` 浠呯伀绠槦銆乣ancient` 鍙や唬銆乣future` 鏈潵銆乣hop` 璧櫘銆乣lillie` 鑾夎帀鑹俱€乣iono` 濂囨爲銆乣n` N銆乣ethan` 闃垮搷銆乣cynthia` 绔瑰叞銆乣misty` 灏忛湠銆乣arven` 娲惧笗銆乣steven` 澶у惥銆乣marnie` 鐜涗繍銆乣erika` 鑾変匠銆乣larry` 闈掓湪銆乣team_rocket` 鐏闃熴€乣ace_spec` ACE SPEC銆乣can_use` 鍙娇鐢ㄦ爣蹇椼€?- 鎶€鑳芥爣蹇楅€愰」涓猴細`main_ability` 涓诲姩鐗规€с€乣once_turn` 姣忓洖鍚堜竴娆°€乣select_activation` 鍙€夋嫨瑙﹀彂銆乣not_stack` 涓嶅彔鍔犮€乣activate_in_discard` 寮冪墝鍖虹敓鏁堛€乣attach_bench` 璐撮檮鍒板悗鍦恒€乣ko_self` 鑷韩鏄忓帴銆乣lucky_bonus` 骞歌繍濂栧姳銆?
## 鏄庣‘鍒犻櫎

`effect_node_graph`銆乣is_condition`銆佽繍琛屾椂 trigger subject/target銆佸崟鐙檮鐫€鑳介噺鍗¤鏁般€佹湁鏁堣兘閲忔€讳唤鏁般€佹湁鏁堣兘閲忕被鍨?mask銆佽创闄勫悗渚涜兘灞炴€?浠芥暟銆佹敾鍑?鎾ら€€鑳介噺缂哄彛銆乶ewly enabled銆佷激瀹冲悗 HP銆並O 绛旀銆佸崱鐗?serial 鏁板€笺€佹棤娉曚粠 observation 璇佸疄鐨勭墝搴撻『搴忋€乷wner-specific zone銆侀噸澶?option 閫夋嫨涓婁笅鏂囥€佸彲鐢卞崱鐗屽叧绯婚噸寤虹殑 option-skill 鍏崇郴銆佸弽鍚?attack/skill-to-card 鍏崇郴銆侀噸澶嶈韩浠?embedding銆侀槦浼?鐜╁ persona 鍧囦笉杩涘叆妯″瀷銆?
