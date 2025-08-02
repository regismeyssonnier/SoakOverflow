import sys
import math
import random
import numpy as np

conv1_weight_shape = (8, 83, 3, 3)
conv1_weight = '''徨彰恝张弡忪廁廰忞巳巊庫嶲帔庭崶嶾废挥挹挷捷挷挼拺拁抴恮恿愳恓怴悺彷彻恙惊悍戜惤愞扁恬愗找惮惂懠悧惬憓怬恊愾怳怼忌怨怊怂忥怳怭怏忭怾怩怉思思忧怗嶖廮巿幺庉廱廍廜巒帇庬廹幍廅徢庺廤庋挟拖懪换拼戫掔拞扱急惚悀惒惚惜悔您恡忝恧悦怈悥愸悗惥愄徿恼恌怼恎惑您恎恇怳徿忣怗怱怞忠怘怑怯忎怏忶恆徽怑忰忊帵庒庇嵹左幟幇帥幪嵳幄巾嶲帥庀嵅帏師悸戤憩憝憁慻徥恠悠廻忀忊忩怤忊彈怖恐庁彚彴忓徊彲幪彷彄嶒庲弁庞庣幙州庎庬忤応忨怵忭徾忶怺忼忝忬忭忖忥忮忘怕怤巬帕巊巁帯嵧幄帆嶃帺巼帥庁巋巗幱巹巘彆徳恒怏徾彰怏忯怋巽床庲庫异帨廣库帙幼帥径庖彞彊廭彣弖巳幧幭庖庙帥庇帬府怃怸怩总恂怠怶怓怜忺怛怙怢忒恇恃怛忴帋幅幐带年庀帥帜店彉弅彐弶彗彏廐庫弽弦彾弇弫忇怚弽怎忉廼庿廈庭廨张庈庻庨幰巶幍巧帎幰帚帙康幗帱巶帣幆師幄幱幵怋忪怶忆怏忄忁忛忾忋总忊怚忊怺恈怦徾廬徕彮弜径彬徃弹怆庁幾廘废幤廢徸廸御慮愨慸慘憂慸憓憻戊戽或戵戩成懑拢戴拓懓懟懀憜應憭扁戬懷怘怡怮忳忓忋怒急忟悜悐愍悡悳恷憞愃愰忭忛怱态怙忙徸怰怟廔弟彔廄巻嶿式帾巬帇巵带嶸庆嶜帗康幃幼嶿巟嶎广巘崑嶝嵙忶徐忳忴幭廿怲帵廤弧彀律弋帨廗弪廸嶱忀恆忹怦忥忼忊怜徽弻彩弾彛工庿彫幞巹忬怺忩怪忍态忬忭性弢庱帯彯幡忘归待庾廑廙廝徾廨怕弙待弡庎帹嵶崤崕崔嵕嵹嶡嵐嶎嵻帠嵷帮帛嶼嶅巑嵤嵙帊嵏幯嵣嶕巍忝必怍忽恉徵志忯忻嵯崱崬巘嵎庀嵐嶿嶻忪怞怿怆徿忷德忴忴帑庽幱形廣庂弃庈庣庞幘廌忖徦忐怏彧彛弄廮庾彋弔廓廷弖府巀帅帐彫庴廽廊巶巠帖巏帠彭弌庶弨席帣怀态怹忄怛忞忈怽忷巧巿巈弑廄幼廼帡帕怙忞忸忺怖忽怆恉忷庩庸廧庵幠弨忎弒引廡弐廒徆廿弻忂彎忂忟怎彉徔彃徾弯忁徹廏庫庅弱彶弭弔廊庺幻弦幫徃廞庿归廃廋性徹忯徹怶徼快性忧廘庝帼彅廳廄弱庅廠忘忶恄怃快忲志怊怒怞恃恿恒怇恧怖怚恱怭忹恲忓恀怶怕忳恽怄忱怘怌忯忮忠徱怾待忛忛弟德忾弪廴怽席巫庀嵣巀庆嶉嶕弤掎推掠捊捝挠抚挲挚悴悑愍忟忿悳彲忀惈恢愖戰恲惨戢悓愰扅悊悽懍恐惾憦怺恡慀忙忻怣怚怼怣怦怇忕怹念怶忚徹忡怰态忕幪庨幝庨庱幏延庝帔庻庶张廀庶弝廷座庽挝拀懘捫拸懅捬抢懜惄愀悴悰恾恄惇忼性恔悚惦忰惂愧恳恌愴悇恎悏恅悕恾怷怬怨怖怗怅徾怞怋怯忬怱忾忏怲忹怵怒徼忍忽巴幤幻帋廞幩巃帘廝嶂帓工嶩帴幌嶣巓帵才懐懟懀懇慕悛悝愂怪德忋怏怟怇忺总恜弸彴彌徃弪弴忎徫忝庀廸弘廬庺弃廤廉弳快忌忬忓怌忽怭忳忒忝忘怞怗怆怞忭忉徾嶗州嶟嶪帪嶩帖帻嶶幈应帥庣巐帢庽师嶶怩弪忻彿弪弯悋恸恤帬帍庎帿序庪弋幯庄幨序徢幽弙弻彐待往店巭幇庎并帩弍庒度恆恇怖必恅恇忰怫忲忉忆怀怽忤怚忬怤怛幢幘庬常庺府帣巡幎御彄強廟彽彉强弥彩弈彮張弬忌彗忞弅弇廒廓庵庻廥廕庌幫庨幟幺床帖幤幀布帠幂帼广帿幗庐康巑幜度忖忣忇怯恋恂怫怐怪怅怟怢怂怢怜怉怙怤弿彰徜弳律式徤徽往廌庶弡幁廃廠忄彈忴慘慠懑慲懃懴懏懛戚懫憭戀憻戏懺抐扙担憼懚憻憚慛憟扟战戚忾德忾忯怮忍忶怴态惩惏惩惽惌惀愼愇愊怭怦徾怕忾忞応忤忘廏徜弔平巿嶥巵庩嶧巠巂巨庤幙巅彁廉并废巌嶘帻帕嶜嶿嶆嶁忭怽忎弡幎并廖度幩弽彴徍帼広弴庨弳帽忕忥怔恊怎忍忶怲忕徔弬徂帽帚帽幝广幛徹怸徾怞忺恁忯思怵廞弱幯弣廻徨弡弾彌弑弝廖忀弉怕彼弾徴嶧差巇嶍崒峜嵗嵰巴嵧巚巔巪嵢帰嶾帇巤嶇嵤嶞帅嵩庯嵛巀庅怵忕徼怕怋怋怞徶忹巉嵮嵢帆嶍底嶧嶪布忢忶忈怋忽怘怆忈忬幋幩廈弁彋座弌庬庻弈庀庑怩徑徝怞弰彤弬張幼弍廹廽弒庩庖帧帥幭彼廻弟廎己巭帩巽庮彭引廚廵幺幪忎怋忯怐恄徺恃怡忥嶐帨工弁廤庤廇幕帑怹怆怰忤怶忁恅念怵庸幘廇廙廒廏徲弈庾彆弄廏彺廲廗忪忈径忋怵彎彐彲徶彳徫征弡引庹彣廈弰弤廕弃庝弐幭弍庝廭廜庹庅忈怪忿忞忧忺怾忰忹廛廹庘彘弙弇彟廑弃忶忨忭恀怘忁恃忍忱恲怫恏怾怛恨忹怖恛忽怱悅応怩怕怉怶恲忩忽恉徏忤怬彶応怤忣忈忰廇弧德弣弲忬帜巕帧巻幚延巄庅忀拞拽挌抿捃挚扫拆挻恥悊悸徜恗惸忾志悤悟惓扢您慐抌愃慇扣惀愧慤悆愜憣悖悭憁忿忂忪忣怾怯忨恊怰徾徹怡怚怔忍忹怩忨廰幍幚巑廂弩帿幅干弅幂弣帏幱庱帿帿廰挄拋憇按戛廩挦拉憕惡怯悺忎忐幺徻怭怚怔悊恸怠忟庽忷恼惯悆怛恵心忳弝彭恏息忏怯怌德怚怇恉怚怸忑恃怲怱忞怑忤忽忒庑廏弚康幰弝彆廯弈巫彊庱帷幁庾幽幻廑愶愪悤憊愺愸彴悌录怓恤忽怦悾悈恵悆忍徹後彾彇忟怚徛怴忘廡影庭廚弴廭度录庚志态忭怇恇忁怦徿忹忘急徸忏怖忲忡忤忂幣幣帝帖庢帥庙廑幱庂庢幷廌帅幦年幛廨彍弪忰従徖忬忺必忤庋异强庘弫彫幤廛庌庻庚怊廹徦怔弶彳忰庪帿廴庀庹式庼幺廕怈恄恇怡怰怫忐怏忣怍怵忷心忐忝忉怋忠庻弋庮廎廁弨幍幙床徠循彜彈徉彝弄弧彜弝徍彨往微徃微徘彦徆彑弄彧彺弓廩廱彑弾弤广廉弉廩席庖弇廛弱弑廗弇弃庿幤廀忿徵忤徽怱忶怪怞怀怃怂恈応怗怯怢忦忉彛彘弩彮徺弹徣彼復廭廋弟弁廓庥彐彪径愒惀憤愩憪慻慐慬憓懐憂戃戦懵懋扩戄或憇戎懣懱憘惹懺户戌总怊恉忇怈怰怹徾怚恠惏悴悚惕恍惧惿惏恊怦忖性心怙怮怎怹庝彬忎彦弻幽平巾幠嶤嵪帻嶫嵳常彞廤床幗帏巴嵍嵵嶗嶰川嵭忼忣怯忼怦弌庋廲庤彑彟御彌徬帶庨库廰怵忮怾徼怅忯忈徸徽征彸彔待彴帶幰幃帰总怼徼徼忯忘忰忈怃廾弤彁忈弭御弯徧彏弁弧廽徟怊怒徏恭怍幃年幘工巬嵙嶿巶应巙州巵幖嵬幩幝帳帄州帒師庒庒弳嶾广廰忨忸怨怠忪忪怠忞忰巙巠帀幌庴弿帼幡序忯怂怔徻忣忡怮忟忱帶庆廗弮彀幷建庴廌廬弔式怄忺必徘徍彬弓弥弨彩弎弙彉弥弇廗帶廭忘徔弅庿庩廪庢幧廀彔徘归彔廔彁恉恋忆徸忧忣怎怰忬帠巹建役彯廒彇帗幆忦怾恁恂忈怞恁徾志廇府徟廸廗廥徛弨彸弽彴弖彯彾彔怋微徥恛忞徦彸忠忏彈忎徝廯弻徊彺弡弫彗廰彧弪弽循彆廣廆廯庅弅念忓怜怌怆忁怘忨怯庶廹徉彫弸廝弓廜彇怛忖怉徵忦忋忖怡态怪怌忶忪忚怈忡性怹復恄怃從忒恠彺怋恛徽怒忉很彽忆彺徿忴惵徖怹恠彡忌悊徝徍怹徲彼怅廸彎愬弥廌徣徊彧弸惏忄徚惵恪慱忥循彧怩彻懮形徉弘彷彨忋怌弊恧後徳恰开徚弈怰弗悜弱強怜忩怔怎怩怣忐性忆怙怛怘怅忈忸怗怷忪恁彲彁廷忖怒徙很彔怪彈弪恈役怊忱徳恧彄悴恕愅彩忸忀悎弽徙廝彦徺彩忑忏徱徔弔彛彷怽徖怉怑徚弍徼廲忑彴廤式弆弰廩忋怭恀怨忻怈怯怳忚忼怹忨怏怍忿怬忄恇怇忔彔弔廹恁彆徬後悥弤悮怡恜徑悌徧恸性弙庡庠弟彂彬恃徳徵怓悏徇怛徻徏忕悁恭征怟忨徵廼徻怪惞忨徊忽德忮彗志必悳忠忈怎怦忄恆忿忯怠忂怉忴怘怗忼心怆忈惊徼徢弫彾弪彋弴忶忲怔徖忁廾强忌弒怟恓恪忛弽怏徐忒彂愁役忒忽彦彁弈徤开忤忍徊彘徴廀廿復强恾忇忀彼復廗弃徟彶恨急怂怍忪忖怠怪怏恃怾恉怩恃怟忕忶怢怫彝怿徤徙忤徬弴恂忆强总御怵怾忣怎恾急徶彏忩忐徲徦忍忺彝强悋徣恄悛忠徛怷徑弼恏彞忮忩忓怮彛怾弝怗彀忢恋徇忷怄徫忰怆怚怢怵忡忟怉怵怗忂忈怏怙怲忯忛忰岦悊忖忍忧感庘忬建廏懪愓憬悷愗弙惄帻慼戤惢慗悞恊役忍戍幱干弼廔廮徭幐幽彁弣徝徶庶徢徚帔幾恚忑忖怦怗忔怳怕忧徾幾忪忩恉很悘库徿徤怕忮怂心忹怒恆徺徿怨忛廻忸師帪悏恤悃憹恟庞憆帄帹憺恹憃弞彣彭嶐嵊快彣徱嶻徾庩廻弾弼巸徢弅弥惿情弛怵嶪弫彋彦怌怓忠怊忹徻怽忴怔忱悶徟幷徶幼帻忌忂总怌忓徸怙忰怑忰恁忐幟忳幬慦捦抹患愬彈徛恂床懁换抖懲憥憁彤怗彭敽擃捈挀描懹帼悕徠拓掗挛戾戫戸幸弬庩拰挚拝憧或慜怕忆恃忪忛怄忘忴忆帆录廮捀搕挅憾慻憩快必怍怀怰怎忚怛徼忶底悅慦延怯很徦巡怆悠惡懝悝慹恀悱幣愹想恲懆惄愳愛慃悳彌怋彫拳微怆怮性庈忟忢惤打悔慗恨恶廡忩徼怵忾忓忙徿忷忛張怮悧択彳愳恞彬幃忯怫恁忣怌怬忒忢快憗悃恞態彧怳慭忯彪懫恨懤憅幙恹憫忿引憾戞慿憦悰慵愵愾恲愱惚慐慱彥忧悢徭当払愆懱憬忞悯愴忹彧快忞徹忒忑忄急怽忧懸悅愭懛徳惰慞怀必徽志忂恅徹恅怐怞徼徃循弾怾怬彬怰怦彧徬彻徢恚忉幪忨忚幯忻恉弰惆惟幮惥恎彅愗御恂怏徺怇徫慰徨息彩恼忻恽恒悔挡怠嶻帮庭巾庠庭徴岛度怡幞幘怊息循怬慑徟忾廚廬弝巭弄廹怚徽恞廼弁弧怌彲怇惏彙怺恄念忳忌怗怜忭忶怉恈怿徽忱忥忳恄怊怌愋惯弛忤愓悼徨彋悗徧忞怷惎忺怡庬序彣彺弟尙嶪廭庸忌彚彡徏彜張廲広忏廩彠恁彩怚忱恦怣彎彂彜恐忴忨很怐徧彌弪徑恄忮急怀怠怀怍忍怓怂怡怣怛怫徼忄忙恆彇懣捠悌弈废态庈戛弥懪搤惍廯忘悜帠拭徟廀式弜徂庌嶠徬序廯懇懼怑微徏怩徉慗徭憈拂愎弝廥忂弬慑弐慑挞応忊弖徾帰慻恅徶忛忕怳怫忻怸恀徻怖怫忋忡忉怄必忴庼弋患戵抙扞恧慁彛律忊忈懽懿扭慖懇恲廉彨弫循忑徥忀忪彍巵庘彏慙憵慘愝憡忔巵廰怸戅懤拚惺懔弱巩幔恲憪戩拜情慱忦忮忛志忼怚忐忰恉忘忶忇怟恂怔怴忳忣忭憬抇戦扂捧扺悙廍徭慠抅愹扬揉护怗徬忍忣憗悰愫懕慔廁怴忾戤抎惸扲挷憿惓彮徾懡拡惑技捃懬徨廊彫抩指憀挧摥抯恹弈志怓応忔忂怈怗怹怭怮徸忦忀忎忿怒忬性怬征忀忈平彋恡帝徫彬戊懗慻廊慒憆弘弰徧慻愭恪幑廘必彎彝忪彁徇彮庵弶忊弎忛悀庙弁廠帧彩忎廕忚徃忍忆怜怏忀怌恊怑徾微怊怵庈快悆廍徉徑怭怚忧忔忷怬怖必快庖悫憹弓帺愵彟惥怦恹拂戾怰幠抉帧抃惡悐惆扨捺愔戶拒戞怆幟忰怆悭弌徍恇怊巕廊惋恣忲式悽忔惼急徾心怹忥怞忂怏忐怬建愮慞德廲慊徇惧忲恇忪忲徺忀怐忌怮忉戡憍懤拰懚扴慖懺崃扢扮挊捾慆掀扎慧廲改撬敚撨晢挕挗揘愎揭捗揝挺拷搂挶打忏拽扛拲拜挛拙戯懵弟怺忛恅忍态忽快恉忏拘抺搒揉拻採戓或幄徸忦忳怵忞怽怇忐怴弓恁得怲役忪彞復弄慤愍悛慷恷愿愞慒悶悔復弘愜恙怪怱忄忚慪慑悩懯惐愌懐慜悰態憓惌战惸惉憻悱惎恋忪怬怗怬怬怴忊怡悥惌怫憣悀慊悭愊恂怭忒怲怌忛急忈恊忨恪意悤愖徫恟愢惉惫恭懠憐愗您慼慦懫懟懺払憿扙戼懚憴憹懮慓懳愕慏怂愦惴惗愚拡抱挚戫戤手戉戆懣总怸恉态忙忞怼怏忧慩抟戤戊愒懿懾懰戋急徻恄忐忿忊恉急忴怺忕忍怔忻忢恝彷彥怓忢彆怑彊彈恘弶忱恑幬恷恅彾怗恧律惧彌弿律徏往德弉很怀巈巓庀巖幅廒帋巾弌挔拗拖拟捑拦拉拙挴恰恟惇怷怫惺忐忬悮怄悅懡惱惵戠惻憀抎悈愍愯惍惢憇悾悻慏怘怅恊忀忤忋徻怙忢徼忔怖忸怟恈怫徶忥嶱廟廌庩庱帩幍帒幖帟庳廸张幑幡幀庙廏捎挰慙捠拿愱拚披慍必忢悹惿怙怾徘怚怎忭忘悋忷怫悝怺怆怿徥怞恵恳怬怀忎径徻忙忕恇忿思怲忀忪忂念态怴怗恃恊忻忤忿幃弲廿应廕廍帖廡廽帍庼廭工廿幗广庫弗悞惦愚慽憫愛怟惁怃徲忞忶恡怛忿忘怓忏弧彄彺德徽徟弿忩怫幑廏廍廠弻庙幰廸庶忽德怮怸怣怦忌忞怙怎忌怦忢怰忚忒怯怖師巖帯帨市幐幠庄廈弄幝庲廊帠幑幣幻庯徽彶忣弚徍怄忖忟怇庒库开庘幵彙幊廃幔廤庣廜府廖応廮弹忑幫并庍幇帺弫幉幩弒恁怱忂恈忾怠怅徾怿徿徶怦忡忲怖怓忪徹庎庫庮廔廦廎序庡年従彮征彪忍徸廪弇忒徜徉徦徙徼忖徘怅忎彔弦彜弶弬彳弎弮弼廴平庿庺廧廌庂幰廲引庝府廈庰廞庱广弌忤忂怪徹忛怎思徺怴忞怵忿必怕怴恁忩怳廃彅徐復徠庍徛彼忈幀帍弊幛幭庩弲弋徰慕惮慯愊慂愸憢慓憿慲憀憿懝憲慗扏懦扻慃慐懾憡懤慍懑憯戣怖徹忥忱忨忹忯怏怗恢悓悴悫惬怀愉惊愴恄徹必思怛总怃忪恁弇彘徿庑帊巪徠幔幛帧嶮强廉帙広嶬廲廹席帻廖庅平希嵝市嵜忪怙徠忋庯床怄廄廀徛徲弸廬弭庾徎店帺忹忱怛怯念恇徼忌怭徣徤弌廢廑幔忶廧幫徸怋怹快応怵怦怡忻弼弞庲徜庤彇彃开廾弡弲廔忂弮恕引弿徾希幅庝巛幂巜帝巼広帇幈帗庩帟庽庞廃庢巫巛庂帶帉彆巗帣帔忬忕徻忹怺忱怤态性差幖巐幱巭廔巭帑帧徻怌忱忟态忓怡怩怪序彐廏廊弊廫彴庈庽弊廧廈徚径怡怑律弟弛廘廗徘弌弓弬弾廪帲庯幱御廩弻從幻庢庉幍年廭弋弜彔康廎怂怡怡怟怪忇态怡怇幋庹帕幹廙影弗幕帋忴忟徵徽忏怰志怭忴廣廷廨彎廽弚彋幢忣彅弳廨徃廾弔怛忳徳恞恋忦徭忏忞彽忴忁廾廞庼弍弞廍当彃彷弍彌廮弤弁廩弐廊廱忖忈怂忣忙怟忀忪忍彉弟弋徒廾廫彊庥従怖忱忦怇怰怢忟怊忤志怓恊怑恈恐忯怑忇征徭彷徯弭德忑徘恆彪彻忬彝忐忕弫徶怘徥忄怋忠忣忌忙怔徱徫怙徱徛徸忇念忓忨忪従忼怣征恅怾忁怮忡怉応応徾忍怔忉怣忼志怀忩忆怚忿忺怪忇忂忋忥怘志忈忻忛忭怣忭忁忧怙忘徿徺性徾徽忿忕忖徿恆忬徰忁怐徃忹忧怟怍怉忣怍忀忩徽忩忶忼応怙忁怈徽忧心忻徰忴忾徛忌徠忕忳徣微怄怖復循徹心怔忦忁怋怕徥忽忤忢怖忰從徨忰徽忍怆怙忨忴怃忷徾恆忬态怟忿怲恁忈忇徇徔徨徸忡忰徭忐忓忣忚忼必徭怑忞徦徶怀怡忡怄徳怒徸徵忺征徯得怩怽怗従徻怂忌忥怓徣徲徐忟応忌志忪徴忨徘徝徭心徵怵忇忁必忉忠恆怪恃志忶怃志忍忲忥忡怰怅恋怍徴忩得怤忈忼忈徶忪忛忋怜怯忣怡怬忯忎怷怐思忨忪彲徺怟怓忿忒忋忘怫怭怃怄念怒徾怃怈徯忩志忲怅忯忁怩忪徫恆恂忺怀怽怾怷怯忛怯急忐怉忘怫忪恀徿忐怱怛忯忇怵怑志徵怅怙徼忚忰忧怪徫忐忉彳怂性忆怋恬忣忾忄怉怴怇怦忮怷怟律後志忺恄忭怇怃态忧忤徳忪怾怛徔徸思徊忄性恄忱怤忢怀怣怤怡念忒急忊怰忥怋怞怪忲怬忆忩怘徽徾徭怪怔徯怯怨徿徵徳怃忮怂忧念徢怃怓徬復忬忈心忀忖応怒忢徛怂忒忯忲微徙怢必徨忠忕怕怉恉徺忠忸忈徰徨怎忥忛徱忢忂怭恁忀徸怑忙怀忣怃忰徽忲御徾急徫怠忐怑忱忱忳必怰忪忞忩忕忭徢怍忍怮忖忺怓怋御怙怒怨忹忙忂忞怞怘怂快徼徽怉忚忡忞怸徻忬忄心忚忰急忇忆忮怖忮徸忯怳忖忱怇忓怿忉徹怘怽応怳忮忨忱従徫思忸忩怆忷忕念復忈忦怢忧忲怲徺怔念怆徃怺忓快徢忩忽恈忚怀怭徳怗徸怗思忝徜忕循忨怖怺忂怑态怼忧快忮忕復忞忽忍徬怊徽怇微恁忹応怹必忝怩总怴忷怫徢忞忺徵怐忙怊忴徾徿怌怎忌忔忌忮思怴思怭忢忎怃忮徾怂忝忆忽忢忀忮怹怮态怂忺徦心忻忷徴忶徽怄怺怑忊怲心恇怺忂忬忉徊忁急怔怼忲怄忁忳怔忧忘怳忽忺恄怘态快怷怈忊忡徸怹忷怫忎忼恉怄怜徨忽怔怒怒怜忓恓忯忷忪心忘忓快忼徽怊忷徼怂忐必怦忋德徾徺忨忴忲怀忦忘恄忤怱忚忣忙忪徽忖忳态怇怵忦忓怎怍怹忆怴忷快忥徨忋忮怌従怋徥徰忩怢忍徊怡徜徵忳忎忿怔怩忰恅怓忐怓弞彚怞廼彻忱廐彚怔嶫嶜幒嶹嶯库嶚嶅廑捷掎捜挛持捭抗拖挂恠怎悋怬忴惼彭徳悐惆惔戅惠愑戰悎惵承恪悾懊恤悯憀怠怐愒态怍恉怈徹忽怂忒忼怗忹忋忄忳性德恁忮幈帮幢弞庻廉廢干庩庩庩弝弬应徜幨幐座掹挾戊挕拽戴捵拭户想恸恬愧惕悬恶怒悑怾恵悦悇恟惏恓息愨恧恥悴恚怞惏怺怖悛忔忙忟忧怗怷怬忕怀忌徽忕怂怍忪恁忬徿巅川廡巀希庨嶺幎忒嵧嶸嶄嵿崮师巴嶐庈愯憄戩懁懊慬悝愘悶徴徚怟怟忱忺忌忦恍弐彨弣异彜弼弮弱彀帆庾廜幆廐幧廏廀弩忩忁忊忦忟恅忣怟忾忉怣急忬徸怰恁徾徽巘巟嶣巺幗嶨帧带嶊幤幰幸庍已嶿帛帤嶺怌弄惀徊忢循念怕忏巇師庴帬帻帗席幧幋庎庙彨彁庚弜徤异廨帡嵼帮幠帤幂庿巹席忑怋怛忓德怖忾志忋忂怛恁忒徻怱怐忠怎巨幄幪巵帿帇庄巪帢廽弃建弃彼弟弋廗弨廡廧廤彣忯廼忏彖弮広廡幱廱庽廛幕庚廷幞帹幠幐幑幵常州帄帻幐巺帾庖席巏席幤忭怒怩必忥怳恉恊恂怭怘恇怨忓忠徵怟急廱录廴弱徝彌彥徼徔幼帴廻底度弑忇彬彿愯憎憓慶憰慭戏戜扝憲憕戻戭扁憗拂戦押憒憢懁憩憋愭抝戗戻徶总忝怪忧怳忝怿忝悳惬惡情悶悪慂意愇忿怮徸忳忩忝忼怖忀弊彬庡廲彍幈幜嶼幨嵩崿巹嶪嵽巭弒弫己帜嵝帬帟崐嶨巴嶥嵮恋怈徜怂忼徖弝序廫很彶彣彎弱廯廱巤库怪怏忪快怩忍忴怇性彌弻征彑彼庿庛巫庇怰忆性怵忄德忧怲怲座廞幜彺座弃彍徸庻庾彐庢忪弝忾开忣彏嵝帇嶡嵢峿崗巬嶦嵮嵲嶋嵨幘嵨左帡席巪嵠嵆帞帅嵹帣崰巟幄快忍忻怖怢快忭怸怶崘嵕崋嶻崨帯嵕巚巈怘怅忥怋怘忀恅忋恂帬庸帩彩底帹弽庞庻庻庢廿怶徲怦恦忉彛廢式廐开庽庳廕弄庿巍巊幟彞座庸廘帝帣巡广幩彘弋庼座幌帀怯忓怐忓总忉忰恆忯嶿嶶嶠弭废幛引差希快怷怪恅怹怈怳心徹废庝弁弅廐廴彖廭徝廙弈座彖府彁忸忆彦忷心彯彖徒彶彇必彞庞廉幞影庋廴弌店弿庘廠幠弛廤廦廳彂幏忡快怎怘忎怏恇怗忞廪延幡徆延弲弴庯弅忑怹怶怐忩忄快怉忸忱怸悊恣恮悈怪怿怭忎恴恗怵怿恑忕恆恥忞忣怃态従快後徖恭'''

conv1_bias_shape = (8,)
conv1_bias = '''惪悺悺徍忕恙彷惊'''

conv2_weight_shape = (16, 8, 3, 3)
conv2_weight = '''徇徦并恔忮异惋希徿弦弪彚弭徜弌復彨徫弩幱徇忛廳怉廪庂往彋悛彃憷悚憎廖廻廟巢弝徻悭怬庩徐戥怰形弅弭弼幁徖悻庡怺惄弼忴恼循惸徛彜怖帳弘彃延庚怅徊弓忌忌廓悉忞廳弴愈忙恂忂忁弙弐彄归德徻惹御很強怠干恖徥怑悸扫揮挟慈据掜抅戡扁挒挣捀捴挭憞掑捚懕廔彴弩彥建彫忺悳怔彣悯徸彥恛怅愪恠怆恜幥悗廗庥态徤徴彭庡廰弚弡弱年忐怼忈态巪巺恶忴帨往弭忟庸廟彎彷徔帤弞役廴徔師弋恗悛徬徸徉怂彴嶂席庄慄惹惂悃忛彶廏徽徻忴彿恲弳廓恺張忍弪徽悦怊悬悛忽巳弳徬延廰彝彶廴怺憄愈心恼愽愩愜怙悇恸慉愃悸惸忓惆恣恧惃惨怹忧惻惐恱惘嵹庖巇帊帕帢帧弲幠嶚廻巁庂座帜弾归师惹忀恍悲恵怊彤恴惛悴弼惒徏悩徂徨弩忍憃憗惓急愥愣忈惚怬徐幻怶弐巍忸惃往怱怋弈徇恝嶺彫慫忮愞徕帾惑忨巖悊愨循怾惝抛懯愁怟惷慿憓憄捒挻捅懹懳括択抰抷忺开恡徜己恩惮怞愤恦忽怒忴恘愘徱彍志惗廛忰弻幑弳怰悢徜慩惛愔恤循怕悽忄惫惓慜愔恌恍惩慝惡恜後徎恗悀惝徤悁惦惋帅弳川彜庆帻帰強帜帑彩弋幋廗弰巘帩式恂愢愓悽忽徔恩怟惝惊悲徔征彊悂從惬徝恰悀恊慸慺徨惮怕忲愜徕慘怦惫怽彾愇忓愪悗怽愱徶愤彿惠慎忥意復愫惰愢悓愍忱嶷彜嶡巺廹幕幥帝弼帲帗巚廭彄庨巤归帏恮悿惈悉彶忱徍怕怪怍悺忛彲径怦惝恜怶恥快情悝忋慟恬恹患徏怮忌悉彿徰恈恤怽悱悄忑徑怎徚彈怊悂恸弹悍徎弭廴怸弈弗忽弉庍彃徍廮庳异性怊廼忕彑廵廥弑怯廭急悪徵忱怭惀怕悛怾弣彯悁彵弲徶恴恴影忒彸忳忕式従彑彩悢惚悫御忦怫惙忘彶悉忷忻徣恒恗恩徦恴愒惄恑愑忺彜怨恩愙急幻徫庹廽彈帰幉復幦廦庍弌弞幵廟彇干廛愬忍怞彬徻悇恶患彖忌愜惰惿悗忿得忡彲患心怏愚忊悊忄徍徱愎惆彾忋恧怌徜悜悱忑悕影彣愱恥愚怢忹彮忄惍忊恳惍恅忽怐帔序帟建帬幩庢廞庿巂己弣廟弓幹幙幼彀惑忑悉恆恔愇怌恐惴怹惦彋恴怱忢彙怖弩愹怗惾徺惣愅彦惗恫忢徔恚徊徸忞惛弖恌廿従忆弿忞徼怷惕悢徭异忧怙异恹恑快总廩幷廡廷徐徣彩忟废帵廪徊库往徰弸徳弸恇惓弲念志开彑怠怄惘弫得忻怇弧彉徍怡彽徆恟怳弧怹悞惮怎徰帐忷廄庝幱恈思悉性弜忥彭废彶往径悒彮廥径弒庘徶恥弤惂憬憬拓惰慭挿怳慥惦扚拚戲捩扻懹授捩捊忘幎弘徤庌恆恙惇循恤彿悶惼彃惒恚怘怑恷帐恺恻庐弶悡忍彲彸弃彝往徐怯惄廿彗徨恿引彵弌彙怣彑徟弮怞徶恫悢彻徿怹延忤忹忇思悹憊恰恈弤弱律恟循悞忭弃強悃徾彜弱彟從惇徺怯恁恻彍彷彺怽徸忉怈悒弱徳忯恐忶怫弤廡悹憘態悔悄慈惪惱慛慽悸慵惬怎怰慪愢悕怮愲怴惪怐您悓怃悎怉弝彉币弚從廡巰彺弣嶪廫强帀幰彧弧徍巪惜惿怜得惩怽怡慌忋恌性徛悰惲忒怼怊愌怠慢怎慝惾忔悆慜悿愛惦怰愔慦惁怉怀恻快慞恉悔怴惑慌徕徤恎恓忿怔彨悙惥徆忖庸幜广庰徏帥忌忴帜废巫彸廙徬廑应庼庛恢悴徶徂彦徯恺彬徫恝怗徤愈忈志患彖惀悪慧恉悱慬怫惻怽慜心张悀彗徏恻彗弽忟忘怜彌忼徹彸弙惀弑怜恢恍悠悦御彄情恟恀心怑怟徵恣怗廖弆幹幙忧廅恔影常怾彛怉悝徺徺怮怩恟恛弧忴悜彻徱待彲怐律彿悡弐惎彺怄悺徇忮悥'''

conv2_bias_shape = (16,)
conv2_bias = '''徑恽快慧恗憔慾怴慇愓忴愙徏愾懫彴'''

conv3_weight_shape = (16, 16, 3, 3)
conv3_weight = '''廡帻廇幷巾庾帉序廷张廿彎忻怖彩总悙恻忶徛廞弿恩彿徱忬徨怐役怫怐忺恆彗彋弓恁怌弣徼廸引恎忠弿徑彚恄忄恖彙忛徰彟彤忪徇彶总忧怺彻彳恘恞彆恕恐彖怸忪弤恇徬怶律彖怴彬徛恥弣彤役弫徛式忉徥怺怫彝恍忞徔怞徟怩弴弛庣忶庮康從怲恑恥彗恠忢忤徧悑恴怼悃徤徳忏徵忯徽恫彦徳徜弩忌徸彌徿怔忌恛忟徙徸彜彜忧彈廘忚我悸慕愋愻扇怗恰憸懾懐懠扬拔挅戲括掯惁惵懙总忲憇弅弡徔徔徫忣态弶忌怳弮忁扴抻戁懐户掳拃挦拥悅开彐悋忤徘惤怬彌彗弛怿弧异忇怛弩忛廗彆徸廙廅恮恎恌怴廯忉徍徱彃徫恨忸彜徭徫彁怓彺廾恫弮怙彫彉徍徚庨怷悕怣慎憾憧戡懪拎抡扲抴捡忽忷总弙怪徫彐彪弘悙彆弓恒忨廞慖忊心悖徹恎彻引怓愱忌恍徉开悓廂彼徧巫弳忻怱怌忯惦徎惶彁徐庑懩慁愚情惽懾恗惷愩愋恓徯惝怴怶怺必徸忄忭恀彝彎忞悡忑怊愅恷想息愸愢恠恌慉悉弩応恘徲廒悼悬怏忀弬彀怣庲快患徦忝彣悧悢弪彘怞恬悧怍徽張忛恾徠恷悱忡怹式徦彨彅庙徘恖弾彵彂忓彧忓彅忳恮忣悐憍憎慘慦惆愲怗怈恽怏徆怇怳怂态廷怇廳怓彐忽恬彉彣恦徃徵徜忹弮怡忌廯忮忲悷康徕彍径忻応度庻怼怏彼怲悼怉彧恤彫復嵮巢嶊帕彀平廀幎帙幾徖忊忆彟廠忌怦徇惉悂惡恁恕怮恾恧悉幰帢嵙巴彿巎庈廞巾惍惥愉惥怐怩怍怂恼忖恟怿您怵怱悫悤忢怵徏恄悪恞恏徤怼悌惓态忣忹惜徭怺惧怷恂悠惆怖恲怃怑悻忾悅悑徟恞彑恄恐徣恴巾巖巫嶥庺嶯帅庨幤忻悶恾恜悓忝忮快悃惧惣怣意愂惧忕惛态惟怲怶惙悰忘惣怦忰忦徟悪悮応恫悢怮彧忏徵彜怊徭往强忿循怎弝廋彠徎徲庩徶怪悅彯怨忐悜往彍恋得彤恚弶心形忩当恪思徖廊怌弮彔彑帾徶怗彐很怛徿彛徊忙忑忰忂役怇怚弹忔彍弬怛恣恄忉忉待忯徠彑徠忦徱彿彊恔徖忷怩忦徖彮強恖忄怣微怟徾恫怄强徃征庋忕恂徹忪恏忳怐役怜彷怔息忎徦恆忢徧忙恲性怞徖廁忀廯弚弳平徂彳悀彎恴徵弰忿彨怴恢忤怙忰徶恦彷忌彨彼彂恠恵念徨忧徕彖忞弄庮弃彮归廷幄废彁弖弻廁弑弆录彁徥廢恕恛怄怗忦忤徱怈怱庴忇幖庝彸廭弾形康怍忋恞忳怅徻恍弣往彞恫彈怭念怆忶往彁徶怿循悤悂忰徃恉恏怐徎彪恍徱忞总急怌怢徱恮恇怈徠忄怩忒恬徼弾恇念怏怦恁怆庼徏庴彛当廆從徟忚当必忰忢恖恜御忓忲役怃忛恈怏弿御忹弶息徫徿怃徰忎弻忄彞恢徚徾恢恩恴忶彬恩弬怛徣弰彵彵忙強怈巂廩幑弟弬幻弇弐帥彈彊徵怉徐廤怍徽彫怋悍惌惛惊忙心悆恣弮忇彀悕愪徏徱意徜忼恔怆惀怰悔忍恪忲情徾怼恰恜悔恓忹忺忊怙徻徕惊恢惈悆怆恕怗徶悃惇忓怆忀惊恐忼怈徹恑徚惞恟恐恗怵徟怱恺怩忑忹恅帳彔巅度张廟巺徊帨悾怛徕徵恹恺惠惒怴忮惎恰忦悥恽悷怘悂忭悼恚怬恱忮怦恎恜悲徛怰怀悌彿恡彵怺徉恣怙怛徬忴忡忴恹弨徃徰忨怓恡怪忀录怎徙徼怕徉怐悠恀彔彂恫悀忝忋徇往役恲彷怋彩忕徘恡忋忋忎彾彲彯怦恂归怏恞彸徶彷弿怡彬彤念怌忩恪微忕徉忧怲恧彴归徎怆忎徣忂彉怀彳忡徇怦悀忙恔怹徴徆恓恮怍恨悑态忕彞怍彲怖彨徎徘弬恩彣怟怳恶悈恈徰怹悓忬徸忰怃怒息急恇恪悀怒忮怰恩彫徘怛恣徜彭彅忸忞忼彳怚怠徙忘怶您徙恺悥您怺循恖忕愥惖慲慃愎憰悑憂慭忮怳悕恵悓怦怅彫忊彫弥廠弎彫廕彡徺彾惈戁悹慶惩懯惖憚恱徝弱忉弡庛後彼恟怸徝廯弹廯彷弌怣弃徆怡忹恭怇念徴怄恣性弡徽彲弴廅庭循彧怌廊序忇忰廐強忰忈忑庻庞庮悇廕彡忉徨徏情恛怒愇悿愔愎惺惘心悌悅怜忠急惖怇彉弞廎弢徿彶彧恫徉廽徂彯彋忓廚徆復怅彷怷态恊怊彾弪忈忋恒嶻店庀巁弟彠嶐巏帜峑帱巋嵔巘幊崑带庪彀忝弒弞忔库庀弉异懃慉戭慷戜愨愡或戃峩弔幊庈応庵幊弛庛憀才愲愗慦扒懛憪才憅愕憊慑戣懘惷扊憥怤忆惉恒怘徜悆恅徔悾懈懁悪戜惹悼懽惶慐慔慖愦慠慀愞懆慦悙惵忯徥悔惡恤徳恘嶮循弰庳弚彡帉彍庶徲悐很忰忎忮恂忝必慏憀抣憳扳懯懂慭懻慘扇払慑戻慺慽懶懔怢悃弙忤怡弬恼忭彭弰忏怄彴忴弲彍忊弋帾庠庤巟彈庨幭干弦怕彟忬徧惐彟恱彯怃惥恏悺怾悴怔恵惶惌嵒峇峝嵣嶜嵼屒嵨崭恳怎怩惱恥恚忙恢惤您惹悁恿忠惨忎忄悌悑悾忕徶怋悑恽恲徬悔恜愈怽恴惌恁思忊恓忥惜惹怅悀悟忲徺恴怍悉徍徍忭徤悙忰庆帯工帯廭帀廴徦带忟怋弌徔徏志怴忬忆恏怽惤悍忒恩惆总怰恎悝愨忰惉愆惹悐怚悸悶怟恎恔忏徬徽悧忾悞怂徖怋悧恱怒悘弇忋忿徉彮怼弲徲怸怬怹忍恼徾快快忕式彐恾彝弞怲徆怷弲忇忂徳怂徿弽弉徥徽廇徂快忭弥徭忔弡廹忂弒悄従徦徃忣徊徫彏役怂恖悐悉忼彾怜徑忷恮張弥快恋徣彷怫影忪忑恕怕彆怯彳彨怺性恛得怩彸恼恖恃怵德怲怈応忸彪弰忷恚彾怩忺彪怩忡怺彼徽怛怭忇怕态彄徛彣怕怉恨弝彞彟待忡徃恻忆忤怦忤従恠徐悄忡悔忹怤悶怷恜怖忓恻徴悘徘恰彪怶悃往徶恤恞徻怶徜怲彼怀彁怦強恕忬彈怜徳復忴彀怐恻怶彖彛彵徭恽徣恞忌恏従恄征怛怪恢忀恺彗徚役忏恨忢归怘恴徵怒徸忹律恋恠恖怇恀彼忎怚徭怯後怎悅彆從恭恼彲怺恖恦徯忥彝怚悋忀徐弻怕彴忯恐快忮弧彦恈悀思很怗悪恔悩忭彦归恰徉彸怂悊恎彿彡怒彡怒彬忌怹彸悜怟徣恺怎彚徤怀怡微怜彣忘彈弣忏徂忋弪怏怕徬徘弱彔当征怅怩彫忮忥思必彟怯廷怪忿徵恅忎恊弴彜弯忳怇徙忔忒怼恜怘怖怭彶徾彨怢弨忤弦怷彀彼弈忔忪彯弣忹彖彤徹怑怳恄恜恄徎彃忟徛忑恐悌彛徠彖怠彪怕徫徔德徢弫悟徏彷忴影恝怨忷忙怳彴彖必徠忁德引忖恦忨恍恛悊忠忳恥恉怪思彻怚彨彷彤徻怮徼彉忸徊廑弟徰怱彊徑徻待忭恴徚怅恸怋徍悿愁恔惜惠悫怕惎恈戔抱懫扉抋所憧憍愐愬意惾惇悀悔彚徃徜彇彖彭忤彙彸恈恃影慏所抄懎慧懀惫戧懦怌彞忁忂彫忂恕恆悯恨忟忞徺怰弭恕悗态徙徛恵恐彈弾彷彟忳恹廲彷恓怄忨忇徼悵弶忘復徥忌弤怓忺忈径徻忲忶弖彂廍弌忛扆戂扥慴扺戜愭憦懗惥恦怀慖慕悬惫恹忚悟従徖徉恒弸恕忣性态怖怪恅恨忹怲悫恧怉彵惊彳忳徜弿恾恾彟悢怾怞恳忮悈恋彿徦怠怯得怾彙径彺恂悝怍恋彝徇悙彧恆恼徆忓恡彨忻恰怕弾怴性怼忱彝忰患忩恪怩弽徭忮弾恷徹恃彪恑待恏徵忦忭徒恩徂恿彬恇恽悉忙怲怬彐忿弶彺彿彙徍彦忔彲徏弯恂忷彬忐徇怷恦息忎恵徲彯後彭弶恄徰恓征弑忹徺彲弼徵忖怛恠悑恺徰悈悈恽彚徐彨忰忩弫怹怃徴徶弟徾彤怭徔怼弰恦彣恚恘彠悙恐徂恐彯悍'''

conv3_bias_shape = (16,)
conv3_bias = '''徉恇怵憭恎惾想徦徜抁慨忛彰弾怭弿'''

fc_weight_shape = (50, 16)
fc_weight = '''戬憖忘慝悒恛庛彫帩怚愧慺恗徔惤惄戙悖後惨悧帻忰忦怺恄忑彤惿恒庨广惄弌慝應慪弌彿弟恩弿懝惒忾帩懬弇幯彻忩戰惼恆惭幃愾怡庲嶳平廍懐忘彃岙崅弽常愻徉庠庆彈市帟徛恶庑廀懕慂彦怕怯惒愘影悟徙惊悉怟库巤忟懪康恿幻憫怿怈惃徲徛恾帪应御徸慷慊庁弖惤怡彆慵慪恩彖惒帅恟快彖慣徊徇彡憂怦悺愢慰愛彈惧慆席忷廷慮徨展嶯心戀愅恬怂彩弪恕巶帕徫帷恙幭惦惼徜帽懩慫懾憪怞彣怨恟惮彘御幃惙忚彟彣怷巠悥廛恵幱悝库惝恽惟徟徝庢惇庬愱廷惐悞悃帰忁彄愾愙愇忟廴徳忲庻廱徂戓得恧弢彋慪廬愜懞帡帨幢態憊忉愉徤怴息怈廝弍廆惓憮悟彽弦念弎復憷廛捶库憽忽忯戕慕廡廮悒惶恣弔廎彁廅抩徣帵得廇恭巾悯嶕慼帜恚彍恄怰徕悰彿幚幤彋懵怣幫惬悐岹彠嵌慝帺帜彇恒恥廄彨忥怑弫忽宄悹座愘怅恵慃巖嵑怸恲弣恄怲愄扦徘巂擯慦抦扮巾惝橖掑怌庖怳懠怂扨徘廥扣弚抯振忢我梛挐憖態慣憤幬懯弢怛抸快懼捠忥彬榺掴彄张忱戼怼弢惖怽掺应忳揻惮张柑揲懥憞懃憚弆庬尶娩嵄悧幫岞幫嶲劋尃愙恲怛守庑庁座愂愣庢恝帪帵嶅彈慆愎廽常愙彸弩惺扊恿帠怇愢归徣彆怠幺惾怊忨憪愜惀愵怿庿彮忑惨幁幷憊庾怌恌徍憧庰憦廔憩徜怓忈悮幧庨廧扶弶弻怀庒态岶尖彜弢恬恥廂已幢彡恤巪微巨怷弞延恴悶延憘庻徒懟很广惢廍彿慨悙恡憍徱憍懗彎廗徴恆彁懣怲悚怱憛忒廠恳忮悰恗怄徫帣悤庴拊巘愵帧惼惁張悡悠徶慢怫恰忯形廧愉徤惯弉憅慘幭峃媅懃廇巤怵幯弶徤想巕幞干学憕恩憪忆崋惾彝嶉恣怓常恉愦廱弗怼彍恤慖悈忝彚怹廍循戱幱怰慸幠弑惶惼憑愺怪库悪幏弙愦憔幸弜惾幘慅惋弴徝惯怨廕帕帞崡恋悚庈幹徃廖忟惮思忏圂屓惺悿愕悠忬嬥愍憲廔廘幕娫懧峙愪嶣峦惶怯嵽怲愭屬彠怅惭徳彁慫嬱愧彃崴悘嶁徲懵恤崮巘愽徇庚彶懕嶲彻庾忌惩岂屓愊抚岕忖慵廵彽彄愦害愛幩幽恔廉廈惉慨履帽恀帰廝懗庆擟布开掩弈挪捱怑徏撅抰幾弓幡嵸廬岶恁忛忥廿岑岸巛扝塱嶏怼庖忂悫徖崸弖度巽息嵐屐憖徸実徫愊彇慚強徿孌庪幏孤应弳嶲怚憖崊寑庛床弒巇忖幃弅怛形怟岍孙弸懣宙庱弄庢悙嵬懎懙幾惸憣弙挑懒徤彃挺搡徐悺廘恞强'''

fc_bias_shape = (50,)
fc_bias = '''徼怚悋愬愘弡怬恛廳廷懗怪忀怒彽廮巍弾庤徂擑擜揹拃姜帥悪微惷庯慗心徐慼怆彘恌惞彴拼廭徾忂庑慵嶓平徺徰愧'''



class Conv2D_Numpy:
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding='valid', dilation=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.padding = padding
        self.use_bias = bias

        kh, kw = self.kernel_size
        # dilation effect on kernel size
        self.kh_dil = self.dilation[0] * (kh - 1) + 1
        self.kw_dil = self.dilation[1] * (kw - 1) + 1

        fan_in = in_channels * kh * kw
        scale = np.sqrt(1. / fan_in)
        self.weight = np.random.uniform(-scale, scale,
                                        (out_channels, in_channels, kh, kw))
        self.bias = np.random.uniform(-scale, scale, out_channels) if bias else None

    def pad_input(self, x, pad_top, pad_bottom, pad_left, pad_right):
        return np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    def im2col(self, x, out_h, out_w):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation

        # Calculate the size of each patch with dilation
        kh_dil = self.kh_dil
        kw_dil = self.kw_dil

        cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)

        for y in range(kh):
            y_max = y * dh + sh * out_h
            for x_ in range(kw):
                x_max = x_ * dw + sw * out_w
                cols[:, :, y, x_, :, :] = x[:, :, y * dh:y_max:sh, x_ * dw:x_max:sw]

        # Rearrange so that each patch is flattened in last dim
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)  # shape (N*out_h*out_w, C*kh*kw)
        return cols

    def forward(self, x):
        N, C, H, W = x.shape
        sh, sw = self.stride

        # Compute output size and padding
        if self.padding == 'same':
            out_h = int(np.ceil(H / sh))
            out_w = int(np.ceil(W / sw))

            pad_h = max((out_h - 1) * sh + self.kh_dil - H, 0)
            pad_w = max((out_w - 1) * sw + self.kw_dil - W, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x_padded = self.pad_input(x, pad_top, pad_bottom, pad_left, pad_right)
        elif self.padding == 'valid':
            out_h = (H - self.kh_dil) // sh + 1
            out_w = (W - self.kw_dil) // sw + 1
            x_padded = x
        else:
            raise ValueError("Only 'same' or 'valid' padding supported")

        # Extract patches
        col = self.im2col(x_padded, out_h, out_w)  # (N*out_h*out_w, C*kh*kw)
        # Reshape weights to (out_channels, C*kh*kw)
        weight_col = self.weight.reshape(self.out_channels, -1)  # (out_channels, C*kh*kw)

        # Matrix multiplication + bias
        out = col @ weight_col.T  # shape (N*out_h*out_w, out_channels)
        if self.use_bias:
            out += self.bias

        # Reshape output to (N, out_channels, out_h, out_w)
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        return out


   

def adaptive_avg_pool2d_numpy(x, output_size):
    """
    Simule torch.nn.AdaptiveAvgPool2d en NumPy.
    
    Args:
        x: Tensor NumPy de forme (N, C, H_in, W_in)
        output_size: int ou tuple (H_out, W_out)
    
    Returns:
        Tensor NumPy de forme (N, C, H_out, W_out)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    H_out, W_out = output_size

    N, C, H_in, W_in = x.shape
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        h_start = int(np.floor(i * H_in / H_out))
        h_end = int(np.ceil((i + 1) * H_in / H_out))
        for j in range(W_out):
            w_start = int(np.floor(j * W_in / W_out))
            w_end = int(np.ceil((j + 1) * W_in / W_out))
            patch = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = patch.mean(axis=(2, 3))
    
    return out

import numpy as np

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        
        # Initialisation uniforme U(-k, k) avec k = 1 / sqrt(in_features)
        k = 1 / np.sqrt(in_features)
        self.weight = np.random.uniform(-k, k, size=(out_features, in_features)).astype(np.float32)
        if bias:
            self.bias = np.random.uniform(-k, k, size=(out_features,)).astype(np.float32)
        else:
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x shape: (..., in_features)
        y = x @ self.weight.T  # shape: (..., out_features)
        if self.bias_enabled:
            y += self.bias
        return y

import numpy as np

class PolicyNet_Numpy:
    def __init__(self, num_players=10, num_actions=5):
        self.num_players = num_players
        self.num_actions = num_actions

        self.conv1 = Conv2D_Numpy(in_channels=83, out_channels=8, kernel_size=3, padding='same')
        self.conv2 = Conv2D_Numpy(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = Conv2D_Numpy(in_channels=16, out_channels=16, kernel_size=3, padding='same')

        self.fc = Linear(in_features=16, out_features=num_players * num_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def eval(self):
        pass

    def forward(self, x):
        """
        x: numpy array of shape (batch_size, 83, H, W)
        returns: numpy array of shape (batch_size, num_players, num_actions)
        """
        x = self.relu(self.conv1.forward(x))
        x = self.relu(self.conv2.forward(x))
        x = self.relu(self.conv3.forward(x))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, C, 1, 1)
        x = x.reshape(x.shape[0], -1)  # shape: (B, 16)
        x = self.fc.forward(x)         # shape: (B, num_players * num_actions)
        return x.reshape(-1, self.num_players, self.num_actions)


def softmax(x, axis=-1):
    # Soustrait le max pour éviter l'overflow numérique
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# 2. Multinomial sampling
def multinomial_numpy(probs):
    """
    Simule torch.multinomial(probs, num_samples=1).squeeze(1)
    Args:
        probs: np.ndarray of shape (batch, num_classes)
    Returns:
        np.ndarray of shape (batch,)
    """
    batch_size, num_classes = probs.shape
    samples = np.array([
        np.random.choice(num_classes, p=probs[i])
        for i in range(batch_size)
    ])
    return samples

def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
	# Étape 1 : reconstruire la chaîne binaire 'weights_bytes' comme en C++ wstring -> string
	weights_bytes = bytearray()
	for c in unicode_str:
		val = ord(c)
		weights_bytes.append((val >> 8) & 0xFF)  # octet haut
		weights_bytes.append(val & 0xFF)         # octet bas

	# Étape 2 : lire les poids 2 octets par 2 octets, big-endian
	size = len(weights_bytes) // 2
	output = []
	for i in range(size):
		s1 = weights_bytes[2*i]
		s2 = weights_bytes[2*i + 1]
		s = (s1 << 8) + s2
		val = (s / divider) - offset
		output.append(val)

	# Étape 3 : si shape précisé, reshape en numpy array
	if shape is not None:
		import numpy as np
		output = np.array(output, dtype=np.float32).reshape(shape)
	else:
		output = list(output)

	return output


class Coord:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def euclidean_to(self, x, y):
		return ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5

	def sqr_euclidean_to(self, x, y):
		return (x - self.x) ** 2 + (y - self.y) ** 2

	def add(self, x, y=None):
		if y is None:
			x, y = x.x, x.y
		return Coord(self.x + x, self.y + y)

	def __hash__(self):
		return hash((self.x, self.y))

	def __eq__(self, other):
		return isinstance(other, Coord) and self.x == other.x and self.y == other.y

	def __repr__(self):
		return f"({self.x}, {self.y})"

	def to_int_string(self):
		return f"{self.x} {self.y}"

	def get_x(self):
		return self.x

	def get_y(self):
		return self.y

	def manhattan_to(self, other):
		if isinstance(other, Coord):
			return abs(self.x - other.x) + abs(self.y - other.y)
		x, y = other
		return abs(self.x - x) + abs(self.y - y)

	def chebyshev_to(self, other):
		if isinstance(other, Coord):
			return max(abs(self.x - other.x), abs(self.y - other.y))
		x, y = other
		return max(abs(self.x - x), abs(self.y - y))


class Tile:
	TYPE_FLOOR = 0
	TYPE_LOW_COVER = 1
	TYPE_HIGH_COVER = 2

	def __init__(self, coord, type_=TYPE_FLOOR):
		self.coord = coord
		self.type = type_

	def set_type(self, type_):
		self.type = type_

	def get_type(self):
		return self.type

	def is_cover(self):
		return self.type != Tile.TYPE_FLOOR

	def get_cover_modifier(self):
		if self.type == Tile.TYPE_LOW_COVER:
			return 0.5
		elif self.type == Tile.TYPE_HIGH_COVER:
			return 0.25
		return 1

	def clear(self):
		self.type = Tile.TYPE_FLOOR

	def is_valid(self):
		# Should compare with a NO_TILE instance
		return True

class Player:
	def __init__(self, coord, team):
		self.coord = coord  # Un objet Coord
		self.team = team    # "red" ou "blue"
		self.last_coord = coord
		self.mx_cooldown = random.randint(5, 7)
		self.cooldown = 0
		self.splash_bombs = random.randint(0, 3)
		self.wetness = 0   
		self.optimalRange = random.randint(5, 10)
		self.soakingPower = random.randint(10, 25)
		self.score = 0
		self.dead = 0
		self.thx = -1
		self.thy = -1
		self.id = 0
		self.idsh = -1

	def move(self, c):
		self.last_coord = self.coord
		self.coord = c

	def back_move(self):
		self.coord = self.last_coord

	def __repr__(self):
		return f"Player({self.coord}, '{self.team}')"


def encode_players_numpy(players, grid_height, grid_width):
	# On utilise 8 canaux par joueur (comme dans ton code PyTorch)
	# cooldown, bombs, wetness, range, power, is_red, is_blue, dead
	tensor = np.zeros((40, grid_height, grid_width), dtype=np.float32)

	base = 0
	for player in players:
		x, y = player.coord.x, player.coord.y

		# évite les débordements hors grille
		if 0 <= x < grid_width and 0 <= y < grid_height:
			tensor[base + 0, y, x] = player.cooldown / player.mx_cooldown
			tensor[base + 1, y, x] = player.splash_bombs / 3.0
			tensor[base + 2, y, x] = player.wetness / 100.0
			tensor[base + 3, y, x] = (player.optimalRange - 5) / 5.0
			tensor[base + 4, y, x] = (player.soakingPower - 10) / 15.0

			if player.team == "red":
				tensor[base + 5, y, x] = 1.0
			elif player.team == "blue":
				tensor[base + 6, y, x] = 1.0

			tensor[base + 7, y, x] = player.dead

			base += 8

	return tensor  # shape : (40, H, W)


def encode_grid_numpy(grid,w, h):
	tensor = np.zeros((3, 10, 20), dtype=np.float32)

	print(w, h, my_color, file=sys.stderr, flush=True)
	for y in range(h):
		for x in range(w):
			t = grid[y][x]
			if t == Tile.TYPE_FLOOR:
				tensor[0, y, x] = 1.0
			elif t == Tile.TYPE_LOW_COVER:
				tensor[1, y, x] = 1.0
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[2, y, x] = 1.0

	return tensor  # shape : (3, 20, 10)


def create_dead_player(coord, team):
	p = Player(coord, team)
	p.cooldown = 0
	p.mx_cooldown = 1
	p.splash_bombs = 0
	p.wetness = 0
	p.optimalRange = 0
	p.soakingPower = 0
	p.score = 0
	p.dead = 1
	return p


def complete_team(players, team, n=5):
	# Garde les joueurs vivants
	players_completed = players.copy()
	
	# Calcule combien il manque de joueurs
	missing = n - len(players)
	
	# Ajoute les joueurs morts manquants
	if missing > 0:
		dead_players = [
			create_dead_player(Coord(-1, -1), team)
			for _ in range(missing)
		]
		players_completed.extend(dead_players)
	
	return players_completed

def encode_ALL_RL_numpy(grid, red, blue, w, h):
	red_complete = complete_team(red, "red", 5)
	blue_complete = complete_team(blue, "blue", 5)

	tensor_red = encode_players_numpy(red_complete, 10, 20)   # (40, 20, 10)
	tensor_blue = encode_players_numpy(blue_complete, 10, 20) # (40, 20, 10)
	tensor_grid = encode_grid_numpy(grid, w, h)                     # (3, 20, 10)

	# concaténation sur l'axe des canaux (axis=0)
	input_tensor = np.concatenate([tensor_red, tensor_blue, tensor_grid], axis=0) 
	return input_tensor  # shape: (40+40+3=83, 20, 10)


class Game:

	def __init__(self, w, h):
		self.width = w
		self.height = h
		self.grid = grid
		self.red = []
		self.blue = []
		self.rscore = 0
		self.bscore = 0
		self.my_color = my_color
		self.IDME = {}
		self.IDOPP = {}
		self.state = {}

	def init_NNUSNW(self):
		self.nnz = PolicyNet_Numpy(num_players=10, num_actions=5)

		# Conv1
		self.nnz.conv1.weight = decode_unicode_string_to_weights(conv1_weight, shape=conv1_weight_shape)
	
		conv1_bias_ = decode_unicode_string_to_weights(conv1_bias, shape=conv1_bias_shape)
		self.nnz.conv1.bias = conv1_bias_

		# Conv2
		conv2_weight_ = decode_unicode_string_to_weights(conv2_weight, shape=conv2_weight_shape)
		self.nnz.conv2.weight = conv2_weight_

		conv2_bias_ = decode_unicode_string_to_weights(conv2_bias, shape=conv2_bias_shape)
		self.nnz.conv2.bias = conv2_bias_

		# Conv3
		conv3_weight_ = decode_unicode_string_to_weights(conv3_weight, shape=conv3_weight_shape)
		self.nnz.conv3.weight = conv3_weight_

		conv3_bias_ = decode_unicode_string_to_weights(conv3_bias, shape=conv3_bias_shape)
		self.nnz.conv3.bias = conv3_bias_

		# Fully connected
		fc_weight_ = decode_unicode_string_to_weights(fc_weight, shape=fc_weight_shape)
		self.nnz.fc.weight = fc_weight_

		fc_bias_ = decode_unicode_string_to_weights(fc_bias, shape=fc_bias_shape)
		self.nnz.fc.bias = fc_bias_
		

	def Play(self, ind):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		
		# state_tensor_batch shape: (1, 83, 20, 10) par exemple
		self.nnz.eval()
		state_tensor = encode_ALL_RL_numpy(self.grid, self.red, self.blue, self.width, self.height)  # (canaux, H, W)

		# Ajouter une dimension batch au début : shape devient (1, canaux, H, W)
		state_tensor_batch = np.expand_dims(state_tensor, axis=0)

		# Passage dans le réseau numpy
		logits = self.nnz.forward(state_tensor_batch)  # shape (1, num_players, num_actions)

		# Supprimer la dimension batch pour avoir (num_players, num_actions)
		logits = np.squeeze(logits, axis=0)

		probs = softmax(logits, axis=-1)
		actions = multinomial_numpy(probs)
		actions_list = actions.tolist()

		#print("Actions pr�dites par joueur :", actions_list)

		if ind == 'red':
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.red):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				if mv.x < 0 or mv.x >= self.width or mv.y < 0 or mv.y >= self.height:continue
				t = self.grid[mv.y][mv.x]
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied:
					p.move(mv)
				
			

			# Pour les bleus on n�ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m�me taille

		else:
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i+5] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i+5]])
				if mv.x < 0 or mv.x >= self.width or mv.y < 0 or mv.y >= self.height:continue
				t = self.grid[mv.y][mv.x]
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied:
					p.move(mv)


		# Correction position conflictuelle / retour arri�re
		# Note: ici, self.action a autant d��l�ments que de joueurs concern�s (rouges ou bleus)
		players = self.red if ind == 'red' else self.blue
		for idx, p in enumerate(players):
			occupied = set(pl.coord for pl in self.red + self.blue if pl != p)
			if p.coord in occupied:
				p.back_move()


		if ind == 'red':

			for p in self.red:
				if p.splash_bombs > 0:
					zones = self.get_best_zone_for_agent(p, self.red, self.blue, width=self.width, height=self.height)
					if len(zones) > 0:
						p.thx, p.thy = zones[0]
						p.splash_bombs-= 1

					else:
						p.txh, p.thy = -1, -1
				else:
					p.txh, p.thy = -1, -1

		else:

			for p in self.blue:
				if p.splash_bombs > 0:
					zones = self.get_best_zone_for_agent(p, self.blue, self.red, width=self.width, height=self.height)
					if len(zones) > 0:
						p.thx, p.thy = zones[0]
						p.splash_bombs-= 1
						
					else:
						p.txh, p.thy = -1, -1
				else:
					p.txh, p.thy = -1, -1

		self.Shoot(ind)
				
		players = self.red if ind == 'red' else self.blue
		for p in players:
			shoot = ''
			if p.thx != -1:
				shoot = 'THROW ' + str(p.thx) + ' ' + str(p.thy) 
			elif p.idsh != -1:
				shoot = 'SHOOT ' + str(p.idsh)
			else:
				shoot = 'HUNKER_DOWN'
			print(f"{p.id}; MOVE {p.coord.x} {p.coord.y};" + shoot)

	def get_cover_modifier(self, target, shooter):
		dx = target.coord.x - shooter.coord.x
		dy = target.coord.y - shooter.coord.y
		best_modifier = 1.0

		for d in [(dx, 0), (0, dy)]:
			if abs(d[0]) > 1 or abs(d[1]) > 1:
				adj_x = -int(math.copysign(1, d[0])) if d[0] != 0 else 0
				adj_y = -int(math.copysign(1, d[1])) if d[1] != 0 else 0

				cover_pos = Coord(target.coord.x + adj_x, target.coord.y + adj_y)

				if cover_pos.chebyshev_to(shooter.coord) > 1:
					tile = self.grid.get(cover_pos.x, cover_pos.y)
					best_modifier = min(best_modifier, tile.get_cover_modifier())

		return best_modifier

	def Shoot(self, rb):
		team1 = self.red if rb == 'red' else self.blue
		team2 = self.blue if rb == 'red' else self.red

		for pr in team1:
			if pr.cooldown != 0:
				pr.idsh = -1
				continue
			if pr.thx != -1: continue
			idx = -1
			maxsh = -20000000
			for i, pb in enumerate(team2):
				if pb.wetness >= 100:continue
				dsh = pr.coord.manhattan_to(pb.coord)
				if dsh <= self.state[pr.id].optimalRange:
					if pb.wetness > maxsh:
						maxsh = pb.wetness
						idx = pb.id

			if idx != -1:
				pr.idsh = idx
			else:
				pr.idsh = -1
		
	def get_neighbors_around(self, cx, cy, players):
		neighbors = []
		for p in players:
			px, py = p.coord.x, p.coord.y
			if abs(px - cx) <= 1 and abs(py - cy) <= 1:
				if not (px == cx and py == cy):  # Exclure le centre
					neighbors.append(p)
		return neighbors

	def get_best_zone_for_agent(self, agent: Player, my_agents: list[Player], opp_agents: list[Player], width: int, height: int):
		best_zones = []
		max_enemy_score = -1

		directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]

		for dy in range(-4, 5):
			for dx in range(-4, 5):
				cx = agent.coord.x + dx
				cy = agent.coord.y + dy

				if abs(dx) + abs(dy) > 4:
					continue

				if cx < 0 or cx >= width or cy < 0 or cy >= height:
					continue

				# V�rifie que cette case n�est pas trop proche d�un co�quipier (sauf soi-m�me)
				too_close_to_ally = False
				for ally in my_agents:
					if ally is agent:
						continue
					if abs(ally.coord.x - cx) <= 1 and abs(ally.coord.y - cy) <= 1:
						too_close_to_ally = True
						break

				if too_close_to_ally:
					continue

				adjacent_enemies = 0
				enemy_score = 0

				for dx_dir, dy_dir in directions:
					ex = cx + dx_dir
					ey = cy + dy_dir

					for opp in opp_agents:
						if opp.coord.x == ex and opp.coord.y == ey:
							adjacent_enemies += 1
							enemy_score += 10
							enemy_score += opp.splash_bombs * 10 + (opp.wetness + 30) * 1000
							break

				if adjacent_enemies > 0:
					if enemy_score > max_enemy_score:
						max_enemy_score = enemy_score
						best_zones = [(cx, cy)]
					elif enemy_score == max_enemy_score:
						best_zones.append((cx, cy))

		return best_zones

# Win the water fight by controlling the most territory, or out-soak your opponent!

my_id = int(input())  # Your player id (0 or 1)
agent_data_count = int(input())  # Total number of agents in the game
stat={}
IDME={}
IDOPP={}
for i in range(agent_data_count):
	# agent_id: Unique identifier for this agent
	# player: Player id of this agent
	# shoot_cooldown: Number of turns between each of this agent's shots
	# optimal_range: Maximum manhattan distance for greatest damage output
	# soaking_power: Damage output within optimal conditions
	# splash_bombs: Number of splash bombs this can throw this game
	agent_id, player, shoot_cooldown, optimal_range, soaking_power, splash_bombs = [int(j) for j in input().split()]

	p = Player(Coord(-1, -1), 'red')
	p.mx_cooldown = shoot_cooldown
	p.optimalRange = optimal_range
	p.soakingPower = soaking_power
	p.splash_bombs = splash_bombs
	stat[agent_id] = p
	
	if player == my_id:
		IDME[agent_id] = agent_id
	else:
		IDOPP[agent_id] = agent_id



# width: Width of the game map
# height: Height of the game map
width, height = [int(i) for i in input().split()]
grid = []
for i in range(height):
    inputs = input().split()
    l = []
    for j in range(width):
        # x: X coordinate, 0 is left edge
        # y: Y coordinate, 0 is top edge
        x = int(inputs[3*j])
        y = int(inputs[3*j+1])
        tile_type = int(inputs[3*j+2])
        l.append(tile_type)
    grid.append(l)

turn = 0
my_color = ''
opp_color = ''

game = Game(width, height)
game.grid = grid
game.init_NNUSNW()
game.IDME = IDME
game.IDOPP = IDOPP
game.state = stat
# game loop
while True:
	agent_count = int(input())  # Total number of agents still in the game
	redi={}
	bluei={}
	red=[]
	blue=[]
	for i in range(agent_count):
		# cooldown: Number of turns before this agent can shoot
		# wetness: Damage (0-100) this agent has taken
		agent_id, x, y, cooldown, splash_bombs, wetness = [int(j) for j in input().split()]

		if agent_id in IDME:
			# C'est moi
			if turn == 0:
				if x == 0:
					my_color = 'red'
					opp_color = 'blue'
				else:
					my_color = 'blue'
					opp_color = 'red'

			p = Player(Coord(x, y), my_color)
			p.id = agent_id
			p.cooldown = cooldown
			p.splash_bombs = splash_bombs
			p.wetness = wetness

			if my_color == 'red':
				red.append(p)
			else:
				blue.append(p)
		else:
			# Ennemi
			if turn == 0:
				if x == 0:
					my_color = 'blue'
					opp_color = 'red'
				else:
					my_color = 'red'
					opp_color = 'blue'

			p = Player(Coord(x, y), opp_color)
			p.id = agent_id
			p.cooldown = cooldown
			p.splash_bombs = splash_bombs
			p.wetness = wetness

			if opp_color == 'red':
				red.append(p)
			else:
				blue.append(p)



		game.my_color = my_color
		game.red = red
		game.blue = blue



	print("my_color=", my_color, file=sys.stderr, flush=True)

	my_agent_count = int(input())  # Number of alive agents controlled by you
	game.Play(my_color)

	turn += 1