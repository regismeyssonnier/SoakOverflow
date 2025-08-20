import sys
import math
import random
import numpy as np

conv1_weight_shape = (8, 11, 3, 3)
conv1_weight = '''恨忄怖怩恏怔恅恝彩怷彞帄弹恛忦恛彣惆恥忌応彡怖恆弝怬悉怫悖愡惦忔怪忑悙往恞怴愾弼恚怮悸悑彈忿忦心恺応徆恤彲忇忤忍強徦恅怕忾徶忛彚恚幅彌幎幬嶵嶃恹彀患彷惆恏彖恳徵悴惑忷愢悻拣抧捆择忟忥庾徿忞怄恒忱忠忖幤忛彍忳忋态悚廦徯崟忁帕恬惓悕悵幗忥弫巬己悉弍恥恤惻忍徢帯惜幂忏忠得忇徹悡弒广悓忯悗弐性恢怖徝徔悷廦彃愀幸徯惹悍弒徺幷惒弰悽怲彤弟忺悉慠庇惘廹怶彍彙恖弿怸忋怒怮徇惁快廲引嶮懭彏懂惧必徹忴恿惞廇怇忩恃徽怔忠忓悕徒徒徐惓忬廄忍怮怳廛怤恖快怋恚徇忳廉怇巶庠怎忟徵忟役怣忖怉従怍息彰微怞恿愦恀恒恶悺徬忟徱惗径慆怮忱徆惑恷徫怖廊恓恾彛惏御幊慂庅恢廼帡弿恟恁总悠忈御悊怠忡廌慫捈帼懤幖懤捆彸忌悅弚怶怞弳忪徐忋忺徰徯忒床心徑恋從廟幋恀恑度惢忢忴怊庶恒异帟弨征彊恲庾怘弜怣徟彌弿恑怖恗怷忛忀恍徵徰怶怦応愥徙恆慙愅忣怲弞恽恈怓恍怫怟忡怼忣恗弃巰巵庛彻悗巛巭彻徎彁怕恸恟恷恚怾怯想戌房惓悕弴抨戊惑彆恋徜怹恫必恘恋弲忮怍徧恕怿廕怎惦廲怂悅弞往怬彨忣徇廼庥幨恛庑嵘恜恄悒悱惩忶恿恎总思忁恈彯悬恐忉悟怫徝悎徾彔惡憂徙恢戏怊忭恍彨恆怨怓弆怫忸彸彬惆峘幆忹徏嶱怌思归徧怟恳径恠恜怴忆忡必捓挕悇怩挂愆忬怛懼徧弻怤忉徠恁怑怣律徹忢恍怂怱恔悍忤悃忢忨惖怪悪後庩恱徝惺忑巸恲庇怵恛彈幸忑志忓彍悿悙徉忢悑怏忧徾怽悊忼归悺忼廹彭忴惠憞彰忬惁惆悰彬忈彲徚復惓彲怈怵愊憞愷憏廩悘怰弻徊徨忼怓怯忰従怦徙悷庅嶍帯寢慷恸庻懝怨忉彁恲忁廰怕怘忬惟惾強怱恰徝怄忸怭慙悐慇怋徿悾恿忽怳廷嶎崙幹庌嵷嵞幤帛弲惱恔弿忷愁恦怟惨悳徑忆悂忕悦惕惄忻怠弳徽忊彍悖憺忣悕怂徥悰怹慅愊慑愊愑幜悽慛廉悭愓怼怼悫恷怚微徟怼忰怰忪快怑工巼座幫巂嵰幻彣恉徍恣怖恠悐弰异彦忰徝恇恝怒悤忼惥恰悂悕愩怲恏悸怽恷愮徇怯徨徒彫微应弿徔徚忡忏怸彾恜徥恥惠悐徽忿恉惈恪恳悧慞恥忼悇怚念忽徎悅悠恲悖恫恅愜惀惎惶恸憳帴戞徚戨恮悅懈您彲徢悕悤怦徕恸怺弿幜帨巘嶘嶣帴弒嶱巛忏恫形徫您復恒悍徘'''

conv1_bias_shape = (8,)
conv1_bias = '''彷彄忠忳彶徾怢恨'''

conv2_weight_shape = (16, 8, 3, 3)
conv2_weight = '''悞慅愇憩惙惻懦恑懄忀惃忑怫憔悒彫悧悤懅愤愂愂憳恍憍慨怹恏恔徟愁憕恞恹恰愾惂慗懇意戎愛懩惴恹徇応徶惿挅懆悼戦扯忝廔忹弗復想惸慆彡归徍忊徼惩忒惼徥感戟戬怩惰悯惡憮憟怽怹憊徣愦忴愳忲怚恾惟懀彠惸戝徿或恮恂慔徶徂怫慅悒恙怍您戧惠懃戶懲愵憐悢慖忮恝引惼憀悈恍憱愇徸巛廼忍復彦徍徬悃庤怏徵忔恡循庢恑建愱慬彬慹愴感惁悴惛忣怡惀庸恹怚怄惂恰惦愙懳恳戣惒憶懺懟愿忧悛恼恠惆悰慐悾憹想憍戭憓悎愈愖悭惔戺抚惴憖挎懦慲扉惚怃愪惠悋悂彫悷徰愁愶彙怬惍您徝患愖庻廽悄愙彼怱惡惺悏忌徼忕怚怐幙慶憦悟得彬幹悖弨思悓惵憈悷怩彶怆彗怯忭彪恆律怯庲廥征幩巹廄廬徫怹幨弜很廆徬懅帖惒徾弡愨恘惯惝愊悯廗康弆怚悅怣愖愝彆憎扊恕扷懰懘戵戯憟惸悺惢徥惰憑恋态恂恸恀慱惧愱慐戒悺慶惆悱徯惥怊忸愅恖怽戭愚恎憪懾愨愵悽怿恩怕彼恟憱愩恧惿恹弬弾幎弢恲微怗惑愁彠徦復庽弃怳彉恦弨忁惰悛徃彻徘廂庳彎怳恳幀悸忴幆恶徾弈帽序徼幥廬怛廄幽徚庂彞彜彬弥恛徖怎必庵很師徼彵常庝廙廛帖廤嶬徦序弢康幞廹怬总徇弌怱廎張怡彧徑徽怊彰悥彼恈怇形懓戮憏憺懡戈戻扩慧恛惜悩怆愷慼悮恦悈愲怲応懫憏慉戆戤怉惛恐恘愹愣悗悶忡忴惞惕恍懰愼懃懳慨恩怤愪惝愩惩恕怨愯忸幅忶幙廢彙忌彃恚录彫忐弨式弎役廞忀怋廼徕慥彙悜惰帺怴忰怚愗弋慖慲恆慑悄廪徘常幆弤廡徴帬彺弔復恥忍從徽御弜彖彖幝废庨弾彟廅弳徵従希徛廥忬忷彆廿廀嵸微怏彎弶忩徎恎忮弯惊彾後形徦悿忾怡恸愅懌憟悻愌懁徜憹憸怰愬愡恽惬怳廙廤怋懢惂憤悋慹憉忯慭惖憇憍悍慧悃愩徣悧怒憬戜惷慭扤悟悃憻慩憰愚悠悸扩戸怌悍憭恰弮悺忢恥悝徐怛怕弹彽徵彻悃惵惦弟忰恪惖憎忳怱愃忇愿怳徤悌扅復愴懊徉徏惤恭愅憵憗悤悝愂懧愫惾愵惖忢怊惇徣忬忩憺憿情扦扑恴恸愳慚懁捭慫慾扅懕扪拮懖怘徨惡恺恛怴弿彮彍彿忎彖忶悝徧忊惟徻弒慊廞弿徲巛帤廀幜幀庨嵸幮廍嶨巫序幟弆徚嶉廣徾廐庨弪巤恽怿悗悟恅慷恒徼徐归憁悯弣愔忣恴恜慓嶹怭忐御廕忸忦彫彅忍幾弡廰彥怷忾忇徻愊怤惓恗怇怪恈志恴戥扳愰户懝戅惃憀愜惥愜愷悚怛徥悁悄彣恻悅徢悯慙慶懋戇慦悃惊怫怖慚恫憢忭忖悒憋憲慅悽憨慑懠悼徝惟忥愠惥彧忎慌忒彇帐忧徹恕徎悞影悝彐恩徿应庳彵弞忱德憢懃憗慻懕恠戽戡慌彰悪憯恎愕恔彇恝愾惈憔悌慅憌忴戟恊意悪忔恪怳慉徊恬徫恰愭懬懾懏感懟悜慿愫惗惟悦怤慸忦愎惂憆帘恜很恅恕恻廄後忼徰弹忳忎彸彛庴弓廵憇懯慸戂惡愽愽戕悠悡慆愓恝悥恫忬怘怞惀恳悄懶愿悾戆憲慥怯愠怱惛悺徹愲忧恆悻恥悷或惮憥懴憜恅怠慑惁惬戤惨悞忕忻帇年巟彌廖悑怿弚怑恇怅徂形怒彙彯忲徴惄憟恳憷慢惱悟懎憃怕愛慆怇惜惦忠彐惆怩悰怫憳慫憉慠戇愱愱恏忸怭怰応恂怕惢惧慆恧恹戕惬憫悂徧慆戺性恘扅悦惜愾恠忨彟恏恱彄強悛恡忤弣悮弦彄弼忔息從忮惖憶應怠悔憨恓懾所忸恃悥恈惣惶彸惍惡愴懇懏悡憂憰怒懥恌徴惛忴悖怟悄彩慳怀急戼憮恔戦截恺悇愤彲憞惝恾慌愉愿憕彛徻幯徤恈悰廤忁彠忦庼弞彑恇廫從忮弘庪'''

conv2_bias_shape = (16,)
conv2_bias = '''忂怦彯怉怸惬恆悌忯忽怤恇恚怔怜恞'''

conv3_weight_shape = (16, 16, 3, 3)
conv3_weight = '''悀怐忥怖彬彵徂悾惜怖悓态悤悅惸悪悱悲弸彃彷徝忥忖徴彗恀徧徫恁弽徒従悂忾恪悯慂忚悽愶怠悳惜悭怊恛忡恉徫总恗怭忼悵惙惹惞恎悪忬恏怱怯徸徳怙徳忍怋忹怓怦悡恲恤悱徘怆恆悀彐彽怷彿恹弹恑徻彰彚彫征忈徳恅従怞徃性惠惁恇患恸悧惈悿恖忏惮悧恉忟忉悊悚惎恛愛忴惻悪怰怰惍怡惠恭忺恙惎徰愁恰忩怟怉惃忦怶怾忲怦悗您恘怨悕惛徚忆忩徇徏悩悀恪徜性忒徤悦愴恞愾惫怘愙愁怹怦忧恀恑惷忰徱惫恆徦恤悇惋怞忑惑悷徱恭怫悰忓恈忞忷彌彔恈応恷怂徯恷彣忸恖怍彬徯彏彴怔彫恘形律悫忷快徍念悗忽怐悲恜患恚惃忨忒恨惨恂恱忽总彾怰怎彠忚恞彮悛彉恆恗恕忹恘忐悂恆恠恠徎恉彗悈怸彦当彔悗彭弾忢悌悑怪怌徶忖従怘忿徳怺徲恕德忲恶忢怓徵忀怩忘忐忊怃怹恿忓慸惛想愧慌怦惶愣怭忓彧彷彻弫忋彧忿廽弴很徝形恒弾怼惱忽惇愫恕愋愪怡悭恊愃徳忪悞怱恰忨恀忾恹悱恞悾慢悦愉愴怷思忎徠忇徿恞徟彛恡恸徺惏怙恲悓忠怹忩征廹待怴怃徕怈恁忳彝彉总怿忨徙恺怰彫怾悬慀怲慖慗恥惘恶惁態怮悧怈恚恻恞悿恨怿愀恘惢思怲愕惁怋恠悱惭悯忊恾恤惜徣慇惒愁悍恓恭惢愁怿慢愍悰悳悘慶惱恒恪惕懹懍憂悪憗憞悡惇怕慊恱怤悐慅悽悹惞廸忑恼微徝忢愛愄惂懩憵慀慱悭惝惄慌惵怗忐徸恮忿怆徎彌怽懐慲慜憱憔慔惙愪恳徺征怰徬徰徰徍弣忖恂愶恴愆恪悺怼悏怚惋恼惎慏憬慚惱憒悒恮恰恿忏怐怦怺怠彂惠憓愹憲愕憃惏悩慚悎憆恻恬憈恥悟惈慃憼慘憗憼惮慪憒憼憚恑愿憌愺惕憡悦慏惛慿悥恾惮惀悈恨惻愡恄恠恵恇悝惀快憈悂惊愔愴怤患悀惕恩恍忛惙忼忪怐怰怗恈恕弓怰忪忡彥恛恒您徛恏慑忳悢愛愌悔恜怕怽怀弬怋彬忇怬徠忱惯愝愁恿怏惾想怾惴徍徎影彂忝循怦徘弌悎惧悙惤恳惜惕悳恇忷愝愦徟怢愛応悶愢悍怲怱悉忰忸彑忈弄愧慔悂恶怗愠慂惶惇恋忨悗応惴怒徼忨恖慎愑恚恜愩想悵恨愀悺悠愔悱惣怿恹愫惮惻惢惮忻愌怉惈愂悮慹憜慰憅懃愎悝意懕懒憜慘憿愂憌我懚悟愕恩愰惉慏愕惉愦悈強怫彆律徕忍忽愵惈抉懕扫憷慤慣懑戭戲忼徚怅恡彊弘徘径恑扬憝憢戂憚憪愚慓戭忠往形忦彟廘徦廳忦慷憔愡憷懗慚懆懂愳愗惜愷悭憸悁慤愆懓惈怤恔悛彴弜徙徢廿扭憓戹戁愙惱扆憴憃惌悽悧惀懇情憝慴愾懋憏愯憌戕懣慠愜愃愵愱惕惗懊惓愦慀惚慇戇慸戜戆慔扣戃惢惕悥徟忕忧惓忟恻恘怑惂悿忂恮徧彰徇悀徕彣彲弰当徯廢弃彏怷彬怓悇心悀徧怵彖怪恇怯怿悠惌悴恜恂悒忢怴惢悎忘愘怬惆忇悻惠徑悳悖忈彴恺惈悷忡惪恥悲恍悍恄怒愇忡惫惡悇恻怣徐愈志忾怨忟怣悭怉忑态怉徐恏惉徲恗徍恋悘悧悉悉悷彫弸彫徼惑恃悛怆惏徵徰得忁惞忹惼悈怶忻怟忢徲忲惱惪惶恼忂忧怷忶愞愄忝快恌悛怹徤悞愽悷悐恠憄恁愦惀愾悀惩慥愁憛悪悧惘恆忟慝怡惥怺悠恓怘您徯恍怪彰怑徲悄悄惾愘憡惊愳悥惇惘愇悖怭徯廦怘彥忪怸廠忧惹惉憙悊恾愔愊愄惿怦彞張恃弡廾彡怚徑恸怶恚惀悲惆愍愩恬忧惦愧怲慧惲惁惸愄忹悪彉悲後忸怗彳忆憛惭悴悁悻意悅愸悍愥恢悛惲悐徺惄恀惓惇愂恕慍慣悜慲悺愹惯恥慍愆慖怕怳惂感恕愋惂愐恈惚惰恧悆悁怮悑悍惃态愱愈愱怭徯恞徇恒徸彲怓怎愭悎愦悰恢悟愠悫怾怆忋恶愦惯恿总悥忣彲忟恙彮徥忬怓怍徃当彼开徿廎徍廚廭忷怢彺得强彧弻恿恼待庙弉怑怃彫悅庾彔惐恇怙恮怉悍悢恧悧忧悰悃愄愯悬愇悠悅愍後徼怔彋彐忾応徕微忲怲忙彆彊弎怗恟恿怮徻怮恜恥徦怞怞彐忻忬恆恞弭怜怶恦徺愄怠恅惹恇惝怭悆惧恑悞念彠悎怍忓恤忡总悱怏忽愆悎惀愮愂悿悍慼惢悝惍憈悄恚忤悪恛怵恴怇恇恒彸怙忞恦弢恚弘恲愳悮愠悜慗惘憒恖慞愈惒怙恄怽恄忶忕徢忠循恿悰恞憜恽慿惡憊惩忀彭彪忮忙怹彣怅忝忎怳心愜忭忚恅恣悵忪惒愉惲惆忴忑惙恠悟忽恈悠忕彙恘忄形愓悏恞慵惈慪慍愾慩悾愭惑惧惤恀悅愅悉憓愚恻感悧慥恭恚愡悊慁恪忬愹悓恾恙悎恹惜恟慁悟怠恽惞惆惟慰愌恓愛慡情懝悾憰悾憝悠惩恵慪憇悥悃恡恡恺情怭愐惡惐弰徿恠忼忯徘悮悴恗慜戬慙懨憯慄憇懽懥忢必忳彵忭廔怑忨弬悵愋悽憄慕憑懆惃悰徔恟志徧徔彖怲弹廢忿愞悡愆惚惂恷惈惧忨悷愠恰恧愡恋愕愆性惌恹怦悝恪徙怊引愷愩慺憈慐慁悾愿悉恜恪惻慔悾悿愝悩愆惦悠惋慂惇憡憩愇惋恴惡惩慕悶憊慂慭愰憊恽恟悽恹愮愐慽悒惀慵惖恛憋惢悋慣愉愼愀恘慞惙悮愚憆愿悘忟必忢恻惷怈悵徼弦怦忩态徺張彵悕忝慎懘惜懋愓慶憦懗憖恢怿御彇徹彂徶徻徹感慍惚慊慺慊愻惉惝徐律快彲怆彤怬彔強怎忡感恳愻怛惬悁惉忏怭怸愞恧恧惧愩慙怪悜徊忕彂忣忭忯廿悰憬悊愜悙慠慣愅惎情怶悱愢愸惩愮愳悳懍憟慢悛慃愾愊憄想愡愽恠恝慻怱慭惺惎悱悽惔愴息悥慒悳恢忸怉忆怃悳恟恈恄怘愀恲恲慎愂悲愡恪悀彺忳廮怫彣怗彪弽廚忢必恸忔彽怼怑怫悂愳慹憊惟悺憖患憈慷恨怌惹慏慇惻恏愇恪恽悰愜悛惔恜憀悪惸恎怌惯恸惬怣悫愙想恓恍怫态悔悐悬愠恠怳怦怀恏怉悥恜悬応恐悗恨悧患情徻怬恖慦愉惲悪惲慷慱惮惊想慗恫怩恸愙悐恾惜惮悝愉惄恽悻惫慜愺愼悖悎悕慄慔慤惧惗悙悇態惐惴恾悍憨悴怺忔律忭怗彮忧彺恚徥恊徘彯彷徟恶恠忕彴恁怍忥彥循恬待怬恦徶往怐恀徠復怐彘徬徸忹恑悀忙彾徽怽徕恮怄形恁徒徎悆恺恝悌忟恵怿彯怪怎悍徸怭悓恂怡忽恪徤悗徥恓恭徢怣恎当恨怩彬怃怩很彴悖徑怍恿彠忓彐忛怵恓忨彷徱悜徼怮忿恭忾恥怳怭录忢悅怵怬忙恬忹怚彙忪怊徦徒徲忘彑恩恱徝彺恺很恬恇悚怮形從悊彻忩忲怚彺徜惨愎憋惨懴憐懪慯悯愮懭懴憱慏惟懰惡想恗惮惊忦悂惼悜性忺弶恵彂忇忕徉忙悩悇愼扊懊懰战戃愩慴憣怺彀彖忽彽彗徥怫恀慟懚懙愝愛扇慷憸懫悈徊徘徻弒忋怨彪徢愼悺愰恜慇愠惐愨愕悛悽愂怲愼惝怬憬愋息恦徨怯怿弯忤彏彠憎憲懺扫懩愛戗愛愚懺憍惧憎惮惘慫憃愞愋憛慣懶慔惁憱慌愅惣慧慍悃慂想憪憾憐憬慦意懣懜愦懎懌悸悧惌悻悰憑憷愍慸慖戣戩惗戥慬懳愍憋懀恌愣悄怒恫恋悈悸惃忂徃恖忚後彤惫惟惠愀慿慨憢懞慇戴慬懀恧弛徶恊忑弖恇恎徶戗扂憉憲懁愝惭戮愋忯恺彿彞彫怔怒徢忣想悅悡悘恝愱悁愙愩慊惋愓惩慁憨悖憇憛悤悋忄悒彙待徊忮恛慄扑憾憰憾憔懂戡憙懴愞惙悯悂慰憋憍悩憉憮憼戅慜懀憜愹懗惻懎慠慼悯悿愱憚愝憤憩惔憅愜慓慊愑憅'''

conv3_bias_shape = (16,)
conv3_bias = '''忧忓徸徱怊忋惨怅徝忁忣怊恸彍心徚'''

fc1_weight_shape = (64, 16)
fc1_weight = '''幄惯庣希怞年弈归懅弋庒彾弍恀憭幝嶯悬庞悮廨巸廸府慾建愹徟忡恧怎悇幞弾憇憋悸怒幵幊慀彬幏延巫廬懡慅彇恷忒憮徕憿弶悂左怖扜懎懗愚扭扉愾廅恲帳惾恠态彀弗幎帪忂怾徭愱幏怳愂廚恟徑庽懍幰徿慧彗徢弱怞徍忩愜廽惄惡憴带懊弓扣庑嶛彖悲恪幦幕忤弼忾建惋懦懞廘懚恀惤惂彇忋惎巻弼彙弟廯弒嶒忄彚戟巳恕忮惝愃徑彵憭幨慔师幘必幑怫干懗悬幹悭忘彌愅惉弹悤徧惤怌恰彰忙懹廢帔庼恦彃幮弔徔憺惩懗感庽幗徤房態懢忧徥庳憋悂巿恕惆彸惭彾庘帿悒恙憥手幕扶扷廚悉恳懓扂憀忏戰底扁彠恩徘忊恣惴怣彳惧師恅憮弛巹弙恷庛彇庠憄弗弳忿忝憇悈張憏愿忧彯帡徘弉慓慼愈怔径廑徍幁庳患愓恫懓怠悾廪弔怪惖帢庀怜忸彶心帨恍帞总弴愼幡席慵彷怣工懒懛忾幹怨慻年憻憼彾恨惣愩庇惉幒廠帼慳懑巴惝徤恿怾徾彵懗憋市巓憚徦弱抬忡憎惆弢帚怠廛憹悘年悓戳恺慬悠得庀愪干慝怓庞懏帵帾弢彄徟幏悢悮懔憍恗惚惫愼怀彴懆庆庠悆愍彻彍庤弖愕彩悰怡憺彎巈嶼惧愊怌廛帴必巌恩工彳慃待懽怇年悞忟忞嶾悀徤廍廙慠忏忮廈弗徤惰帩忩悁彁巧弥廡庞戂廲庯岣愁庻崌廜峥忊扟徺屗嵤忡庋忻忦愙心懾慯忇庍慍怯弎应弅惛徯憯底帢帎幸帜恡幘庭惌恾彮懏帯律忽幹恿悮慀忧幉悅恋弶戞憰怜惍戙怴巔懔悈彃幙嶬忐恓悤廪巕慼戺憇悊归币憆廆憛恉彔府帟弑恑惏彭惘惲廖帣憈庠憣嵥岦帗忇座尌嵌帔强弪惠弐廰慙戹弧彟悶恥愱彎恚徉懋廃惗慕恝弨戸徵懿慛彥拥怛怪帆慔戼惍憕庉恸戡彴幡庄忎慰怌愪惀忓庋幦惸怫懠惉憠怭慎愡愊憳振憣恖惩抂扎扂庪惴拴惰康忟惨悷怣微悫幫弤慝忠恒愩录悂彰忴恫庎愩怤惪悃彭庍怲庳庘弼徲帢徿彩徙彀庥幤惢戺庵徊嶡庎忾惽懓徍庆扤彘彲廬扏扜彿従巯恒德扄悰廳戊惏懮懓愣慤幢庝恻愡归怬愑愕弙式愸愛怋慵慙影憮廝懂幥幯忊巅怴弸廸嶝徿懐惺悞惦憖挹慍抟悢拝忳彛我徦戤扃愞惜憐庳忋廻座恰弫廨恧愈憜庬帘幰悈憒怃幇徹庍愘徰憟庺幧廲戍幤幯憂憁幡思慊慤怉忖恬庎彎惓扗怚惨憴懖懶恄懲慦弐惨忘慑悞悮庄年憥怹徦憻怵懣愣幸帎庥徾忱彂弯懀忊忒弲惽彝恉忪懻惭恘希彛怐弇弫懰幛憯忟彇总弈延扱捔心挌怿恡庹很懧懙恙御徾恏怎幘惹徼愧披已慓憰悜彴惷徹愊悳悺慽慗徇弼惎怍幧怈帛忊恛恘怖慌慊忩怷憟慏拖怂拘憔恚惱彠扪戩恾徜怒捯憒惢巻庪弶戢弙彃怠惀弇憰帒憓庠彍廏怰愀憻总弶憡帅德愇帋惩幨幆悶徍惼幫悡懐廣庮愘慫忒幂幱忦怺怖开帢忙忨幑怛恦当悴帤懔怒惌廪懯憁慙嶳徶年怒忹懖恈彸戉彝拓慱扌愌幓扫悫怍彽念帰弟常幖彄悗憴愐彘彑忳態帆愸慲恽忋憕恔廞彟幖徱惿廍彏懡応徕庅愜悔恼庬慴巶悋怒愱懐幟従悳幐廢懏庒抉廃怕徙慚愂庫憄愶戺怋帒恉怡愃希廔恒悓弁愇必往徰懞彔帄憁悋惼'''

fc1_bias_shape = (64,)
fc1_bias = '''帏慭愕愉微愦所帯扚式徵恉庥怍怕恄愳惟懋憢忾异憳懗憶憿愫幃彯怹慘廨憒怈徿憘忟幹慄懬廋幡扛徕徫慭惙帧惠干忛惠幐循愯弹徇恿恒帬慺忂彽廱'''

fc2_weight_shape = (5, 64)
fc2_weight = '''情忧恰悤微弿彨恖彥彦徏惊悧忼忌怮恮总恢庹悶忹怹征徠廻帤怐悐徥弚悔庚往悏怟愦忙廛廪慫庍廍憢恔弚徔廀弾怬悂悮怏惡忥忼恶忬怼悳思忚憍恁悱忧往悏惆律庲弎廰怅悦復愛惐彳心忸律彰往愑恀恩徳恐彅巨必徃惸彔忿幗徺愜恲慛徍彗彝愣徐廋惇彖彪恚弈弄忸愗恪弳怀忷恱恍忆忷怼录忊愤恴惋徯恲惇怍怓庠快弐忓恳忂悫惈彣応彜徆恧彴惠当怂彝忋弴幭悐彴恋廸悵廪怌惿志愘怛微弤恦当怊憘归廰怨平忁徽惠徫彂愙廘徵彲弙息忝弥恰恖待悲忍彩惚悰弯徔忊忬惆恳惢惼惀怙彠彰怀彆延恡彦弦怷弅徥巎忔恾恌忮彆庫恹愅怳愚徧弩廚惂怑弙惿彌弬恦恧怜悭悛悏徥慅忾悖忰庬怰彃息怤悖忄悬廷忾悆悿役廷恤廩從從恋慀悊惊惔徾弗彭役德徊悡弊弩廠幟徰怷怼弊惆帺怗慟彆悹怱弚忹惂忻彽悦微忠悥徑怸恒悛恾悉慍恐悖忇弱恮恓怷徾怲彵'''

fc2_bias_shape = (5,)
fc2_bias = '''忊忉徸徿忕'''

#blue

conv1_weight_shapeb = (8, 11, 3, 3)
conv1_weightb = '''徖彟彩強廏忹忊忧怗弬忉廳彶怕廴弦怇復慣忈懇憩愨抎恹慅愸悮恠怱恕情忥廳怑恷志惚恬惹忀怯形怹恟慘弜憧憫愊慄忭慐愻忞弮徐廧忱忈彥性很惸恖徵悁悿弐徲怾徟悋彝得恢悷弶彌忣徦廇懓彨愪嵅崋彟庰悓廷忧忨彂怋恌廓弥彨恖悾弱廽怸悉怋忮忚忾怭帷彙怋徫愶弜恒慮彍扐戺懡忽幮幇帕応急恧弯彭忉惑彐忖悙悔忧徃弚彋怙徃怺愄徰戫慙愋恟弒彑徫彮怬徨怣怦开岴庡弘慙愍弖弜御恶惏幛慝彉影彲怩得彸悤恉您惋慞幁帬嵽捞抗怾掰忟怗徥徛忥徝徶恣忞徼徎徱徃庣怌廠彩恕忕恮忂延彆恎忴悀庯弇廟庒惱彾嵥意廵庼怣彠忼恔幍彚弾忇帙徧從彷恘応怉忟徬嵊怴怟怩惵廻怅悰怳忾得徼徇徬忒彀忔弝忐徰忨徕幈庁徾帙廔彘恂悻恛得忝彃形彇恹扇拫搄憙弡捏愡摖幮悞悱悘忁彤怪悅恉心怘怛役弊悈忝怙張怶徶徤怣幜忺忌怡庆忋忲彖愨惒悹怅忖恒怬忻怔弜怼彜恏彧忇忀总強愚忣悄悱徶彧恟惄懃戈恮憥懈悯悘悗徎恆怖彦彶愤徊忷悻徎开廎廖庋庶弌幊庨彘惌弽恍恊忷悢悈彼愨惉扒开懅戂憏所扈慅悬惪悞惚怒恫悕恢恓建彁彘怊律怂徃彷彴崰怪忽忌恿彊愃悧惚战慘帟恪托惫帙恺徬徣怠忎弾徻帮従忽怉恻怲徧忋忕庛彎悁憏慘慾恋懅懻情恎悷忀徙怣忓恏忠彑彺快怟弹弳恋念庢忯悟怈恔怀悎怎徤忁徣恜徵拞尹拻摅揑悰徶换挘忐彟忏恟忼忬悂彧恩恿悈怡徟徱恮忢弙忕悚怪彺怮忔彊恗忢徜悒庌幹忐帩怭幎座恕忌徔徒恨微廫彡恠怇恗忏悗愐怯惺恉恚怏徵憎悲恝慖徣愠悊弜恑怎悢悇惋恠悮恾悚巃怭嶜崼巾怐广崓嵠彟恷必怭徢怾悑恩惄挎惢揢掃搓恡懇摘掵從恹恖怲役彥弫徣怬徯悊忉恉悀忚徑恠弁悾您念悽弛恰恋怗忳庪庻廁應巄庉懏弗徹怅悏怃徆忛怚恝彥徐怅怺忈忋庅忾怆廪廢忐嵉帒愤席廂愔帺帆崤廒御悤惮弉愍忻悛悿悵悷彯怣惊帇惴帖悎怽怉忀忘忭忥忝怶揰憛抟巫忑担崰捰役怚忤徧彴律忦徫怳怌弤恅引惩忼忨弩忾底思怂形徐怫彰恒巿性憫扃扚慱懃扼愰截扔悸念怟悁忲忥恤弱怲恣怄徂彽念後循徏弭慺感憚憉慃慔愻戅惷廑惁志弗彏怟怍恝怗悰徂忒悷恗御慚廌悏彯徃彗彄悪怍弸恤怌愹寣弒戽憫彍慶幨愘彼彼彡徵廼微怕彛怣'''

conv1_bias_shapeb = (8,)
conv1_biasb = '''怷徯忺徧忄思忽忒'''

conv2_weight_shapeb = (16, 8, 3, 3)
conv2_weightb = '''徻忇悒悅悛愫悤惫悛恍弬悪恤悤惨弋怑惏悐徨悌弹悿恃很愰惙底忳忩庛徏怂弾徶快恐悀悤惡忙恮忲恂愢扞慘憤扌愙慍怗憢慃廝忑弘忱弗忎忾彡彬忪惬徢彋忄弌怜忰従悸懍悜恓愫忭忑恜悋悌憯愀惡惜慽恚憖悚愼悁悿惷悁总愗彆怖忸徠恹怳悄怖忹徦忹忠徎恬愎惝忏恽惺惲巀帄嶴帾嶧帯巬巟帉愆恊惫慦懏愄愛悍愻悠憮懰憼慊慐恆懆懝志忻徶悡忖忰惍惔忑悝惿惥惀恧懟悫憥懁愠意愊忷忲彽徳怐息徻弎徘怩弝忬怱彋彘悗恦悹悂怂恉快惡愓帚巚帹廥帝廼弓徆师悗悌憈愫恙懀恻憶慵慼悞愮惄惠悤怛恆想從怦忾悭恌恋恒恗惋恲恟帢怊慡恎悥恎憱弳惢徠彑悠性惰惾彁彿愇忍忟怵微忧弄很悡彇徧怜忻惀慎惲惰幊弭徂庲幍彥師庌庩廲忟怳慗憔忍憋憐悗愡彃忠愯愅悻憒愛恥慷惾慹惇慆慑怉慠忴憄怍慀悛愂憜悝怪慥愄恴悳怬悅怘恍恓惋忹怦彧怚悑恨徂徯廪恜怏快愉愂惁恻惊恭嵽嶗庪巠嵟幹帼嶵序恡懓憒憑憝惪慵慍愙愂懏恲恞應慗懵慩憦恙惦恨慯愦愽怴愩悇惩怹惬愀愓悉忌忐彪恂怺応徰恭彧徕徢彩影忔徻廿徍怞恵弻怃怇彁怍悯忑忬总惠徾彻录庩廛廬庫從弑幛悢懬惣慥愇憡悑惛惛怉惕惹悚惶恪憃怛慩忮弮归恬怙情影彼恜張幘并徙怫忧平幷庥彼恡恜惲惕悐弼悽悋恨恿恎徯忿忌惫怆悐悼惂忘悭愤惁恤惯彥惻憚愨愺惾慝愇憴憕幫帶彰幉已帆延弨巖弊忌恘徼忳彶忑弢强悈庴廜忤恂悰悋恭怲廼徬忤弎彖徿当恺徨恛恹愜惩忎恂怡忩怒忕悘徰悮忇悁怐愃恉悢悍憥恹憐慺忣憋徳抃护懆拏慯憻惾戊懲幫庆庫幀建底待巐帉恒悇怆怓忈惻弖徽忹憛惤扪悐愾憸惢憂怃慜忢惴慅愒怓彰徥愜悋悲愅悅慡急悮悬慻悈憄悄恍惫感恁悯惙懳戗憎惣懞慱怤息慖恧悸徎念怗悙弒庼彟憎憅恓懩憕悺悮彗愤惗恣憬悬憮慪忑惋怉怦彲彣愁恽悛循忻循恎怚彔怡怚悀幯彌徏忑悡忺悋愒怫忱忀徝愎悌愙怀惜恺慕慕悪忝恙慯愿愙慪愷悛徼慽悉惣懢憟慌悠愩愻庩弹嶨嶫帷库弬建幧恶志徛彍悴惶怤张张忨廽怀弻役彯恩復愐彘恚态忹忑恍彘忈忴忕恆徱恔悼恄恳弮惎廂廩弇怅廪廟恊廬彊惢忯悴愜憋惏悭恾愂慹愱恊感懊懾戭懮慥徎恃怫庂庺弒弘從徿恃悤徬怂御惾怣录悰慏懟彄憻恝徴怫恞恼徭愵廕愌愭御慨彖懠徔怺愷徴恬彣徛恆恘惟惞惓強悃怀怭彏您悔恆忘惨悐徺慝愮愣幙嶩年帷廔弾帨幎師恮惚戩怤懀忣慜戀戵扔思志憵情愕慕恖悾怣憾悿慎惬怩怿急怅恖従愈悎怩惽徱悙惦惭彘忇徰忛彾彣忤徳悌廷応忆怽庫弬恠弑徭弦彫忻徛徬怅悰恻徫庶彝彃弲廓弪廾徍惻愵慄戀惛戜惸愕悀愖怰悾忙惤惓愝愝怑彜從恁慂愈慪怹惫愧怯從彾恄愲忋懧怩扁悹愑悲怣志怐愍惖愨愪愃悡惾悐愽态恳惁惑恨愈戁恕慥戽慰悑彩弆廬彸彳庤徑建幣想憇惕憂悪憭悢戲懝後忳徤悏慞悊惍憴慙徘悫恛庼徻愴恳怟惛徻形忣愌悃忙張忨德彩幽廣建徼彊徳恘弊忼彀幞弫從彰悲彖彚廔彟庮弮徺徸彉彏役悛怟恨悉恷彘忆悳彎憑愒忾怢恸怴恉惸憃怽忷徧恗悄德忞忦庹弼怽忼廉悤憀弉悘怽患徧忁怩恻怎忘徏徿庑強弎御弎徥彁彍怢彪彣幼弝弄忆恞恎忹廼弓廤彑恧庭徯弫弌怸惵恀惓徸徟徜徳彇恩恿愣忕悞恔恼惀慫怜徑徸悘微弹恦徼弤'''

conv2_bias_shapeb = (16,)
conv2_biasb = '''恒徎悄怖忖悄恜徻彬悙想忕愞彳慝慾'''

conv3_weight_shapeb = (16, 16, 3, 3)
conv3_weightb = '''彿忇彉彊廁徼忬忌弼懧扄扮慊戍扱懘懱憉愁悾惪惎悷惹悞悿恆慀怅惱怗恛悕恀恲悻惔愭憄慙恹惪慤惇惘徠悯惆徢忚怋悕徭恸忬怪忟怜弉徤彸彚彘嶜庨師庹度庐廚庌幔憺想悓慉慊慠慏慿惸復彚恅彾恷恍彔御怣彭庌彜廁忷徿忙忻彴抒拄拻懿戵憹憅所抳徐律徖悃彮彾忢恇恼憒愆懣愿愶愜愗悻惹彾忆弢怕弌彚徍徳怃弌徜弑彿徫徻怍怈徂惒悆怓怆恝悬恁愔律廿廛彌弞志廸弎怒忀徴怓忄彳彠忧弡怚怀彑忂忛廹廭彻弼彺弫庘彡廬彿徻徾弳序弻忄後徣强怳忎弍彲怃忞忶志徼悄怎恡必徇恳恸怼忹愄性悾惞忀恁怑廢忮彖廀恥弞广怾徠徏恏悅引悁悎怐惲悊愇愒恓惭恬惭忨庩幢庡弛廾彤弈彻延徛忰忙忒徙忤忒怘恱怤廙府忪忉廾怓彂平忁悚惌徏恂忥忺悛惉性怔忺徰怫恑徛忮怏愩悖慐憸慬憻憍愱愫彠忢廽後徴彠彤廭幅彑引徬廰彇廭徇忑忢幰彘弻庬彿徍干幻弘徘庛庶廓怣從弋庯廏底弋徦彔後忞弎恛忁悴惠恂惱悊悚怌怑忻懒慕懋戅扅扈憕愘懿悘惁惌悿惤愇愲愹惬愰惉忘悾怗怌悍愇恑惭忹悗念悶愃惟惒恏很弃廱廒彣彠庇征彃弼役怅弼廌彍心弫径恕慌怨慟悥急怰愯恲弘庿廼廻弦廷恊弉彷従忹怆廷廅徠彐庽彑弩彐役廚彣廆弚当廇扒懅憨憴戰愩戄懷戏怜惒惨悝徥惟悖惜恾慘性愝怘愓恎惱愃怑惐慒愞慫愡恦慔恚憧徧惐怯惈從恆怑惊徲弥弿弤弳弇彁怔怠弢弸庁廽廚弙徸幘形彭慂惦惖悫悇慯慙憫惟怺恆悈彙恈怡影怘必庖待幣廅庳徹必循徆懌懭抋戵憾憭慟憸房惈恥恢恛徣怆恰徘徨悤惈憓愧憅愎悐悖恾忞彫廮彼彳廨彗弁徧廽廒彠彂彅廳忡待忒廍弥廱当念弰徫形役懖慚憡懟悯懻惬懗懥怍恜怤惸悑忶悝恃惂恟悦慳恬慥悁慀惫愶慰悽悳悘慔惭憝慧慗怨愃忟惼惄恋惇忶怊弱彟彦怎忔徣弑怶弪幏廧忂弔徰弓弥彵徊惨憃憎悄憅悛惗悤您悜忘悅忷徒怼恈徉恉庍徜彅廦廦彷彝径彿惺愻慣懕惠惋惁愚惵您忞怛怴恖悰忠悠悄悸惺愁慷感愯悰惮恕忊忟徐念徔弔怩弡廵徻引徠忌廏弭忤忷庴悙恬恱惜悑忱悍悒恏徴弬弨忤弯恐恮強彼悬悚忬忑怬徱念悶怾忻悝忀性恍恚悝惥御忴怤忚思悭悈徧怇徔惚恛惓恲怗惜怦恼恦忣恠徏忁恴彖徦怄彲悌恻恢忼悋恷弞廊徿恹忰徺悩恧悑彿悘弖彫彰徝怀恂恔弑律弜徐悮怸徨後徼徿恠态怗彇息悹彡恗惁恑怊忤恟您怱悦惰徬悐惛恃彀強怤弽怋彟廵怳怒彵忥悇悚怒役悌恋徣怸惇恽恗悍惉恌怒庁庼彜徾座归徟彭弹憝惺懶愇憚慑憒戀愠悫律徶悟徾忈悭怔忄惙怏惈愬恍愍慁怟惧憅悌愗愃悴愾惝憢慁恧怢徤惥怂忇忴惎惺後御怖徦怞忿彙弯彏幒弣徉廕床弦彂弟弡慛恁憢愘惗悚慗恉憚恿怇忒式怔忎怣怗徬庲庱庿庱庻彮彘彴徕慂憙惎戽憁懴憚懚戉息彨忭悑影徯怱忣怔憗悼慌愼悂怗悰悻悂忲忳廣怀徦廩廞徆廣忥怂彧徉廫徫庙徰廱惆彫忸惢徂忦忉恟很怦怶忾怆忂弧恎怨忂恑怓弼彨徛徑忉忯怛循弱得徏彫恿庍彂弽廚徣弮彸忲弙廿恔彞彚弪怗彛忋怓弱徹彑忳怄怣恈恮悟忉忨恹悏怓忷惪怵悴悒愊恽惗怘恷忢惲愄悶怫慆悮怶息快徴恩悸悓悅徜怮徂彪忝忟怄怦弫庬廋怢庯彑彧幦康弣弍怒廅律彐忁廦弃廚怶愍悘怎悠怌恹惤怮怯弇彵忆弫廮强廠彐弛徱彽彷徢彵弟彶弧愄忷惷悰恑愤悷惆慃弋廲廹徕弒廪徢徱幩庹忁弩弝彨廚廉彝廻彷彎庡弴廸忖廠庛往幚徠幖徖怇彺廦弽彘弛弞往彺怆忨廏弘張情恦怎忪忪意怐惚忀戚懪愂扜慫慝抎憏戲悖患慮愼悙慒愔悹惒急悱悔悝悵惑恠怓悷悫忡怣恀徖怔怘忮忭廹弅彘彀忁弔币廬庑廫彋廅忖彮怖忰怕彐悯怠愴恃惤惯忼怩恬徟彔往弱怐從弿廤怭徟廨廍弤廃度得彾徺情忾怌愵怤怭悷慄悙庹忛式彞彔弃廐廙庺待忚怑忚怂廚徖忹异徉彥廈庆彷彴并弣循庬彩彤开弈彚彖忍影徍彽忲忓忷廡怈怗徘忕怓忮总惃悮悇怭惎戭慳愯憑懽慃扑意戇怼忴恈愍惘憇悙愗怭怊惙悗怾悭恤悕愂惿惍悾恡惹怏恂忷怬快庽彄思廪怣忊庸廽彋彺彖弘彉徚彷怵廝忢悡忿忪忳慏悬忳忮慗怌徍彲廪忮徖怉忼录御怗廷弜廧念忼异廴悓恎悷愁恌惑愮愃愝床市弫廰幞廎幋彙廩弃庿弭彵怐彎怐弽弣彰忑彊徊忛忽弔廨形廣快忔恊当役恅怺弼式录弻徨怣忊彯徑忲忧徺惪恰恏忎忕恕怓悮憚愷懤懟慀戱慼慱愛慇悴惾悷惟悼愈悺忟恳惊御忲怆彻徍怭怕徼恭悪悡忐恒徸恝幔徜忼廛弊忇廙彠彁弙忋律弟恂忯徝彆恃怙慰悪恏悄忏怡恶惄弱廨弒怎弃弫彝忊忿怜弋弝弇廗廼忪彨徐徆忛恞悖怅往恆怌彘徥徴征悉悛徯怠徉徝忁忟彫忢怼徜忽悁從恏怤悆恀怴悀恿忉恅恙忻恘快彴怟怈忠徜恻徧忴恫恗征恽忼忦悘忚彪徲彶忒悃徵忳忉悈彨怅徻徉恤彮彬彟彮怊忇怜恍怋德彚彝忪彬恂怄忏徝徍彼律怨復徯录怷彉忀徜恑恨怪忩忭応忩忁徣彨怖彳悆忮怜忻当恜忑怦悑彬徣怗忩忎忡徖恍忡忙恁恫悔忧彬徂恇忲忩応恌恀恿怗恱慤惜惙感惺惩悦愛忇廴忂恙弿弑弞徱怕惫惧惞忷恱惷怸忣怈恑惰恉怔悼恉恙恸恟惼愝怌悕恹愺惍怘惋恀恫恒忽愾悜怮怴怀您徾愌怅愃怰忼徵悍慔慉慜慙慁慲慭慐惕忣惤悥慎恺恸悂悢悢惀悻怷心悜愊忐惝忚愧愗怀恦恦悞惮忼悜弣很彽彤廜彏弿廡忥恿必怟恲悍忤悟惖悽惁愫快性怨恑恝怂悎怓想恍悱徥恳悁惙怩忹悓怭恰忚恺忪悰恜徥忓悌怟徺悫徸恺彜愐慞愋懐懐懰惁憖惍恰悷惜恖惻总惞怐忡惃惆恩惫悦悇恉怭忷憾愘愱悦悤憡悉愦懝恉快息惑悉惍恧怇惷徾影恮忼徜恣忣悝怨庴弰廴徎当彬弻張弰惁悛悒慴恴恼悜悛悀忔悚怌徣彽总恨急悎彿彧忪徹悁悋彜悮徬懹愐憋惑懲慖愹慾態悧慁怙悪悳怉恻悾悳愒惹惬恤恗怓愂徽怷忽惍徢恨徵徫怫恖恦悖怲忖德徿恹忌恆悲惼总恒恪忒悈恙情恥廙廑徖弾影廷廏庽幡张彊怿悄忢忄怡弫徃弔忻怄弋忠忂幎开徊彮忺徃廲弔怕并廿忿廍忋怓弰後彏廄念恛恑惉徨徚忍悑悸徟悹徽惣悟恍惶悙惤怢悉惁惊恆忒怖愡忀惋惝徴徔悦忶悽怮怚总悥怌怴忼恒彾忙恀恪念徑彥怛忙忰影廭建徻弓忨徯彶彽恟怤彝徴徢忆恠惍恏惈恩悘悲忦恐忄悚恽強怑彔徼律彘彻悓恔怨恽忑弪恎従怣忀恌怲恈悱怜弞廻幺弨徤徍弁弲弆忓弿怔彫弴彋影弿忛彂彈彞廦怔徃庎廯忁幀徐幹怩彶彴彐徶徂干徘弖廊彚廂延徯弎循快悛恏恶怆悋恭恅愺悲慢惞愼慞愓憐悀惂悗悁愎慭憯惦憪悐忼忡怴惂悈惠悡悋悠怈恎怙怕彰恮怮弧弞帔徫徨廨忬恺幂弦怪徻影彬弗廲廷怦弎彖恚恪忌忐悈微彝恧徬怇徧弜弳彬御徝忎彮忙怊廘怴廌廨怦忌忔'''

conv3_bias_shapeb = (16,)
conv3_biasb = '''彾彧忊恥忖徾忓恭恲彺悌彞忄忿形恠'''

fc1_weight_shapeb = (64, 16)
fc1_weightb = '''廖庍抚巧戡憔巑惄忔捌惺忉悦废序忱愛惂悱悋忋恌息心廉愄恸愢彑巕愙慾庫懥懴悯彭廓徤惋廏庮悶徚愣幨廙廟慲恼忓慘悸庒憭幥嶥愸幣怛彈憂愴惔惡帮庒弢徙幅廨憞懈惴废怂怒庭悛愵忱忌惶彐彧廟惁忸愄彃幙廊惟彝懅帎惟帢徼忊惰悠平憀愮慼庮慗怄徳总怰悊彧憑悁巐慶工慚恤悕恽怈恚悦忐战懸恫怩戒徖憈惣御惸彬快愚帘憻廘弎巔彉応悈彩庿怜彆愳戅怦彷懥庘悲弁廻庽抂忲幍忱岔慴憑抔折惎恀必恌抚憍弳徤懐憠慦惉慾憆急巜怍徸彏徲庤憿憘愡捲排怼捇幩悉岼廊弅憴拆帇帗恊徃平幫悴廵態忨庩忰忷憕帣想忣忾帄彚彛怿怺憏恄慕彚慩性懺徵廗幓愶慱憪忹悕戎帲愝悥悾恑徹慳悳怉怴幕懒幵憫庞幧徢弱惐惹彍愧忔懴录忭怽弻庙恖强悠徸恪愸扴弜忴恄恟忁庽怔弔彤怽惃惰嶱愎庣弢惁忝庐弱帘忰慄庂愤愰庂忈彾慰憕徥忋恫怀彈弫幱嶟挋恬庋扑扁幮怄恾嵎嶇廘廲怼彘很怤恧惋怭庋廍悛怳帗弳怆恤廭師悞恹張意悉忎恱幃愐徟愺忌广忥愒幷忈憞弲庤彻愎悖彶慾広悾弎愅惍彳慢庋怩弄嵨嶾抧当忺扤彅彿扪怣抿憨怦庈悕怤戤忸幪弸懘憵懸慅悐徹忋徭彄慗巧巃恍慚恁惙忨愸愂忱怹庐忐帿慾弡從廌嵎忯忣恄廣憀弇扴托拯憪愖惾廸懓慴弱戔得懄怼异拙彨忆帾廑慵慵戢幨弝徶愧嶲戺懼徘悸总息嵆帨愿惥戦峗帳怸彙悆弟快彬幯怠懚惴廈恤彸悥庆戛恟忧憾底愢廑弗弧忯愤慃恟库惧忱愻廖悎恇巩愱慛庋徣恮愫慮慖徑憾幄廦惩悺幢悺悞慥庥待影幑愼帹廵憗恀幤愛抑愦忏怽戈抄并庬彄廤廾惚慽惃嵪慬忤慴彥帜懌息惱惔廌幻悯庈徑帋憀幑惿憇很恦惊恰悮悸恞帬慯幖憸惃庽廙弽帛怮憥徰廖恰恮忯弇彡應忿帿彿弫懞彺戵怆巺抹嶗悕引慑忬怽師忾工憸忹干帅憙弓廎序恟憰庸帗彬弧惂愺恥帯愱憗幀惼帡懂彔怷彥愠帶広心帢愑慒快幘幞憸怪师愥慥帲惕憊弄悇徂廦引悡弚弸恘彵庽急戼戶帞慤愓廘戁慩抵忛扊戈弼彞徥愠怹悿幛幈恍惼惉廊嶫慮惒忛悺忼忣懌彌恹幚憾幏惵怇扯惶彌忱恛恊扌彬惠廛庵恠幯彯患愰憹币帹憘庶忰愗忴恫惡弹懢怗徽忴慁嶰徧忻崈悿必忖扆戹悔憥慶愇引愘懯嶅嵽扦崪悽憹徟彣悛弜悬徴忄悐怋抡彂彊帯愝悛廯悷弾帩惹庢慥帉憿师愕影幁悞憳懹彍帵廲弴帧徛幟帖惣愊彚態廣彎戣思循庐戅悃憅恱惤慳彲愀慖恏懝庻幀帤憘庈弘庾忂悂怯忩惷帏彛廎弘廟得悼帆幻忪幾怿恙怳懥忂惽愂承惋嵶挒戚徚憌忨愊庺悞悤慌抷帮怽忐帩抪帚幩惥帘怸手拀拕帔戊徝徵彝拺扐廎按悿彑拽廦嶈廂恥幅怙扯憔嶝庇忶懥幁慍庾弩帬德彔庹忙弗悼恦弨挄弪展抎恘慴憌彾席屿廇忧忼愴崜廖徼弫忼悴惀怒庵廄悔恗愅惚忁愨忴慘彠抶庺廈戞恘徴弬忺廤徨幢彊弋愽徻忋庤慃巭帺已徻恘恝廑慫弥惂慱幇惔弣惢徕弉恫懛徊慢幙悫廸懠廊忾憺憁幮庅憞延徃忮嵺怐恄挤戻幠悶慤慄怬'''

fc1_bias_shapeb = (64,)
fc1_biasb = '''慔弤憒憒廜帧幙愊情惙憰恟忮徂弨廩庚怹惲悀径徝幻幕憨慿廅慄悲弈悗悲延懈惴彗彥弯憛憘忡廾忱慨怪慏已懣憹廭廸彯庘廞復惗懻幚徇幜懵帕弝悯'''

fc2_weight_shapeb = (5, 64)
fc2_weightb = '''慒徜惎徃恛彈悋從得怇悚弌彍彄徫廐惥惹忈徏巬恃很忀惖恖怕悔徯彶恟悱惀怙忚怠悇悚忢彅悓恁悪彯態弖恊徥忀恑恗忉忮彍幺慜廌彀庬徍怉徾志慦惇悴忕弈彑恦彃怍徍恐愕弞庳怢彔怗徼愡怱彽庛彷怸悸怍彊径恞廲幪恇怉怿快当恽恱彺怀恋恛恁慄復惇恊惺愝惝恶徼悲忚惄幬愊徘彑嶚恙念恛恏怮惥恨徘怍忣思恾悇当惝愧廘店忠恰弨恐徟彟忊庅悖恛恞愃彔怉愆彑度徫忺心弉怏恇徳忳徻怯必忂恷怎怆彙恩惁恬忠忻弶悾怚彈惭彜徱庮恈徴徘怘您愢忼忄怑怯忖徺怵忱忪惀彿廯彄恎徬恰怀恍惆弁忲恋彦悚怲徵惱忑彗愕徔微忉彅悳徭惋彗怹忙怾悖弸恝廿悉恤惚応恺怂彼徢店惉廬彷帋悁彶怦徑愫悈役怞徔怍惁忀悵強悱悋恝廹径忟庽惭恣彛忍廒忾彬彩恳彅恫悑徶幒惰恔彳快徂怍悳惋役徽恊彺惄徲忉廴彉惂悴恎忹怯惈徬庽惴弇恴店徠徴徂忇惰'''

fc2_bias_shapeb = (5,)
fc2_biasb = '''快忲忛忻怃'''




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

class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.gamma = np.ones(num_features) if affine else None
        self.beta = np.zeros(num_features) if affine else None

        self.running_mean = np.zeros(num_features) if track_running_stats else None
        self.running_var = np.ones(num_features) if track_running_stats else None

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        x = np.asarray(x)
        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected last dimension to be {self.num_features}, got {x.shape[-1]}")

        # Calcul mean et var sur tous les axes sauf le dernier (feature)
        axes = tuple(i for i in range(x.ndim - 1))  # ex: (0,) pour 2D, (0,1) pour 3D

        mean = np.mean(x, axis=axes, keepdims=True)  # shape compatible pour broadcast
        var = np.var(x, axis=axes, ddof=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            # gamma et beta doivent être broadcastables sur x_hat
            # gamma, beta ont shape (C,), on reshape en (1, 1, ..., C) selon ndim de x
            shape = [1] * x.ndim
            shape[-1] = self.num_features
            gamma = self.gamma.reshape(shape)
            beta = self.beta.reshape(shape)
            x_hat = x_hat * gamma + beta

        return x_hat


    def __call__(self, x):
        return self.forward(x)


import numpy as np

class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Paramètres appris (gamma et beta)
        self.gamma = np.ones((1, num_features, 1, 1)) if affine else None
        self.beta = np.zeros((1, num_features, 1, 1)) if affine else None

        # Moyenne et variance courantes (estimées pendant l'entraînement)
        self.running_mean = np.zeros((1, num_features, 1, 1)) if track_running_stats else None
        self.running_var = np.ones((1, num_features, 1, 1)) if track_running_stats else None

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected input of shape (N, {self.num_features}, H, W), got {x.shape}")

        # Moyenne et variance sur (N, H, W) pour chaque canal C
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # shape (1, C, 1, 1)
        var = np.var(x, axis=(0, 2, 3), ddof=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        if self.affine:
            # reshape pour broadcast (C,) → (1, C, 1, 1)
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)
            x_hat = x_hat * gamma + beta

        return x_hat


    def __call__(self, x):
        return self.forward(x)


class PolicyNet_Numpy:
    def __init__(self, state_channels=11, action_dim=5, max_action=2.0):
        self.max_action = max_action
        # Convs (in_channels, out_channels, kernel_size)
        self.conv1 = Conv2D_Numpy(in_channels=state_channels, out_channels=8, kernel_size=3, padding='same')
        self.conv2 = Conv2D_Numpy(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = Conv2D_Numpy(in_channels=16, out_channels=16, kernel_size=3, padding='same')

        # Fully connected
        self.fc1 = Linear(in_features=16, out_features=64)
        #self.fc2 = Linear(in_features=128, out_features=128)
        self.fc2 = Linear(in_features=64, out_features=action_dim)

    def relu(self, x):
        return np.maximum(0, x)  # ReLU

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.relu(self.conv1.forward(x))
        x = self.relu(self.conv2.forward(x))
        x = self.relu(self.conv3.forward(x))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, 32, 1, 1)
        x = x.reshape(x.shape[0], -1)                    # shape: (B, 32)

        x = self.relu(self.fc1.forward(x))               # (B, 128)
        #x = self.relu(self.fc2.forward(x))               # (B, 128)
        x = self.fc2.forward(x)                          # (B, action_dim)

        return x            # scale output like PyTorch Actor


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


def encode_playersTALLNumpy(indp, players, players2, grid, grid_height, grid_width, game):

	tensor = np.zeros((11, grid_height, grid_width), dtype=np.float32)

	player_a = []
	
	player_a.extend(players)
	player_a.extend(players2)
	limit = len(players)

	index = 0
	base = 0
	for player in player_a:
		x, y = player.coord.x, player.coord.y

		#  vite les d bordements hors grille
		if  (0 <= x < grid_width and 0 <= y < grid_height):
			
			tensor[0, y, x] = player.cooldown / player.mx_cooldown  # cooldown norm.
			tensor[1, y, x] = player.splash_bombs / 3.0              # max bombs = 3
			tensor[2, y, x] = player.wetness / 100.0                 # si born    100 ?
			tensor[3, y, x] = (player.optimalRange - 5) / 5.0        # de 5   10
			tensor[4, y, x] = (player.soakingPower - 10) / 15.0      # de 10   25

			if index == indp:
				tensor[5, y, x] = 1.0  # canal red

			if (indp < limit and index < limit) or (indp >= limit and index >= limit):
				tensor[6, y, x] = 1.0  

			if (indp < limit and index >= limit) or (indp >= limit and index < limit):
				tensor[7, y, x] = 1.0  # ennemi
	
			tensor[8, y, x] = player.dead
			score = game.rscore if player.team == 'red' else game.bscore
			tensor[9, y, x] = score / 1500.0
			
		index += 1
	
	for y in range(grid_height):
		for x in range(grid_width):
			cell = grid[y][x]
			t = cell
			if t == Tile.TYPE_FLOOR:
				tensor[10, y, x] = 0.25
			elif t == Tile.TYPE_LOW_COVER:
				tensor[10, y, x] = 0.75
			elif t == Tile.TYPE_HIGH_COVER:
				tensor[10, y, x] = 1.0

	return tensor  


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

def encode_ALL_RL_numpy(indp, grid, red, blue,w, h, g):
	#red_complete = complete_team(red, "red", 5)
	#blue_complete = complete_team(blue, "blue", 5)

	input_tensor = encode_playersTALLNumpy(indp, red, blue, grid, h, w, game)
	return input_tensor

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
		self.nnz = PolicyNet_Numpy()

		# Conv1
		conv1_weight_ = decode_unicode_string_to_weights(conv1_weight, shape=conv1_weight_shape)
		self.nnz.conv1.weight = conv1_weight_

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
		fc1_weight_ = decode_unicode_string_to_weights(fc1_weight, shape=fc1_weight_shape)
		self.nnz.fc1.weight = fc1_weight_

		fc1_bias_ = decode_unicode_string_to_weights(fc1_bias, shape=fc1_bias_shape)
		self.nnz.fc1.bias = fc1_bias_

		fc2_weight_ = decode_unicode_string_to_weights(fc2_weight, shape=fc2_weight_shape)
		self.nnz.fc2.weight = fc2_weight_

		fc2_bias_ = decode_unicode_string_to_weights(fc2_bias, shape=fc2_bias_shape)
		self.nnz.fc2.bias = fc2_bias_

		#fc3_weight_ = decode_unicode_string_to_weights(fc3_weight, shape=fc3_weight_shape)
		#self.nnz.fc3.weight = fc3_weight_

		#fc3_bias_ = decode_unicode_string_to_weights(fc3_bias, shape=fc3_bias_shape)
		#self.nnz.fc3.bias = fc3_bias_

	def init_NNUSNWB(self):
		self.nnz = PolicyNet_Numpy()

		# Conv1
		conv1_weight_ = decode_unicode_string_to_weights(conv1_weightb, shape=conv1_weight_shapeb)
		self.nnz.conv1.weight = conv1_weight_

		conv1_bias_ = decode_unicode_string_to_weights(conv1_biasb, shape=conv1_bias_shapeb)
		self.nnz.conv1.bias = conv1_bias_

		# Conv2
		conv2_weight_ = decode_unicode_string_to_weights(conv2_weightb, shape=conv2_weight_shapeb)
		self.nnz.conv2.weight = conv2_weight_

		conv2_bias_ = decode_unicode_string_to_weights(conv2_biasb, shape=conv2_bias_shapeb)
		self.nnz.conv2.bias = conv2_bias_

		# Conv3
		conv3_weight_ = decode_unicode_string_to_weights(conv3_weightb, shape=conv3_weight_shapeb)
		self.nnz.conv3.weight = conv3_weight_

		conv3_bias_ = decode_unicode_string_to_weights(conv3_biasb, shape=conv3_bias_shapeb)
		self.nnz.conv3.bias = conv3_bias_

		# Fully connected
		fc1_weight_ = decode_unicode_string_to_weights(fc1_weightb, shape=fc1_weight_shapeb)
		self.nnz.fc1.weight = fc1_weight_

		fc1_bias_ = decode_unicode_string_to_weights(fc1_biasb, shape=fc1_bias_shapeb)
		self.nnz.fc1.bias = fc1_bias_

		fc2_weight_ = decode_unicode_string_to_weights(fc2_weightb, shape=fc2_weight_shapeb)
		self.nnz.fc2.weight = fc2_weight_

		fc2_bias_ = decode_unicode_string_to_weights(fc2_biasb, shape=fc2_bias_shapeb)
		self.nnz.fc2.bias = fc2_bias_

		#fc3_weight_ = decode_unicode_string_to_weights(fc3_weight, shape=fc3_weight_shape)
		#self.nnz.fc3.weight = fc3_weight_

		#fc3_bias_ = decode_unicode_string_to_weights(fc3_bias, shape=fc3_bias_shape)
		#self.nnz.fc3.bias = fc3_bias_

	def get_MoveI(self, x, y):
		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		possible_moves = []
		occupied = set(p.coord for p in self.red + self.blue)
		origin = Coord(x, y)
		for idx, d in enumerate(directions):
			new_pos = origin.add(d)

			if not (0 <= new_pos.x < self.width and 0 <= new_pos.y < self.height):
				continue

			cell = self.grid[new_pos.get_y()][new_pos.get_x()]
			if cell != Tile.TYPE_FLOOR:
				continue
			if new_pos in occupied:
				continue

			possible_moves.append(idx)

		return possible_moves


	def Play(self, ind):

		ARG_MAX = False

		directions = [Coord(1, 0), Coord(-1, 0), Coord(0, 1), Coord(0, -1)]

		occupied = set(p.coord for p in self.red + self.blue)
		self.action = []

		
		# Choisir le joueur courant

		player = self.red if ind == 'red' else self.blue
		playerb = self.blue if ind == 'red' else self.red
	
		actions_list = []

		for idx, pl in enumerate(player):
			# Récupérer les moves possibles depuis la grille
			poss_move = self.get_MoveI(pl.coord.x, pl.coord.y)  # ex: [0,2,4]
	
			# Construire un masque booléen
			mask = np.zeros(5, dtype=bool)
			mask[4] = True  # "ne rien faire" toujours valide
			for i in poss_move:
				mask[i] = True

			# Encoder l'état
			
			state_tensor = encode_ALL_RL_numpy(idx, self.grid, player, playerb, self.width, self.height, self)  # (C,H,W)
			state_tensor_batch = np.expand_dims(state_tensor, axis=0)  # (1,C,H,W)

			# Passage dans le réseau -> logits pour ce joueur
			logits = self.nnz.forward(state_tensor_batch)  # shape (1, num_actions)
			logits_np = np.squeeze(logits, axis=0)        # (num_actions,)

			# Appliquer le masque : on met -inf aux actions interdites
			logits_masked = np.where(mask, logits_np, -1e9)

			# Choisir la meilleure action valide
			action = int(np.argmax(logits_masked))
			actions_list.append(action)
	

		print("Actions pr dites par joueur :\n", actions_list, file=sys.stderr, flush=True)

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
				
			

			# Pour les bleus on n ajoute rien dans self.action,
			# ou on ajoute une action neutre si tu veux toujours m me taille

		else:
			
			# Actions des rouges uniquement
			for i, p in enumerate(self.blue):
				if actions_list[i] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i]])
				if mv.x < 0 or mv.x >= self.width or mv.y < 0 or mv.y >= self.height:continue
				t = self.grid[mv.y][mv.x]
				if t != Tile.TYPE_FLOOR: continue
				if mv not in occupied:
					p.move(mv)


		# Correction position conflictuelle / retour arri re
		# Note: ici, self.action a autant d  l ments que de joueurs concern s (rouges ou bleus)
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
				dsh = pr.coord.manhattan_to(pb.coord)
				if dsh <= self.state[pr.id].optimalRange*2:
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

				# V rifie que cette case n est pas trop proche d un co quipier (sauf soi-m me)
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

game.IDME = IDME
game.IDOPP = IDOPP
game.state = stat

load = False
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
					if not load:
						game.init_NNUSNW()
						load = True
				else:
					my_color = 'blue'
					opp_color = 'red'
					if not load:
						game.init_NNUSNWB()
						load = True
				
			

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

	my_count = 0
	opp_count = 0

	for y in range(height):
		for x in range(width):
			best_dist1 = float('inf')
			best_dist2 = float('inf')

			for a in game.red:
				da = abs(y - a.coord.y) + abs(x - a.coord.x)
				if a.wetness >= 50:
					da *= 2
				best_dist1 = min(best_dist1, da)

			for a in game.blue:
				da = abs(y - a.coord.y) + abs(x - a.coord.x)
				if a.wetness >= 50:
					da *= 2
				best_dist2 = min(best_dist2, da)

			if best_dist1 < best_dist2:
				my_count += 1
			elif best_dist2 < best_dist1:
				opp_count += 1

	r = my_count - opp_count
	if r > 0:
		game.rscore += r
	else:
		game.bscore += -r



	print("my_color=", my_color, file=sys.stderr, flush=True)

	my_agent_count = int(input())  # Number of alive agents controlled by you
	game.Play(my_color)

	turn += 1