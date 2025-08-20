import sys
import math
import random
import numpy as np


conv1_weight_shape = (8, 83, 3, 3)
conv1_weight = '''恩愫愿惌愩惽惰惬惌廕忐恾彲忌惠彉徢悦恓怋怨怣怒悈往怋怦彿徧悬強怊悒忚忳悌徘応恀廿怎悁弻徬恎復忏悱彲恃愕律忤悬怌怰忛恀恄忳恋忕心怎忯怷怆忓忌怯恁忕恹恫恗怪总悉恷怴忯怪恻愽徤怘愐忕恮惉彳徫忀忿怔忷忟思怤恙悆惃恙怾悳恌恪愇徿悎悴忡忛愨怳恣惥徎忹悺彣徉惓徯怞悖怞忘怞怨怗恅応恉怷忍恊态忢怿怘必怲怗忏恣惚悕恫惥悥惎慯彨彏怴弶志怒徧怗悐惜悭惈惶悱惍悶惠悓张忧怫彳很悜弾忥惺忰忸惟怅怚愮恚惚憃役径恲徼忺悫忞忳愞忶忕忯忡忁忈忨徺忎忢志怖怿怲怀徹怠忆徭彗弩徜廽弧強度库徯彅彅弁廉弘彐廿弾微忎忆志怋怠怲忼怊徇彰弄廬弶彤弽弝彦彡廗忄廯庾很延弧彜弜弇徃彄庒弧廾廫弢忀徽怊忄忴总怩怪忢徼念怒恂忇忴徿恀恃彬彑忒徲彤彊彰徹彮彎彺彥彰式張徍彴很恗恑怦恪息悔徘徐彤徐徏弝彗径弓彄徍庉彀廽弑彸弅度弤彯廟弳弘廱彪建建彎弣弅怬忢徽怫徹怤忐怿怭怸怭忤忷怆怫思怏恉憨懱慛懗懒懅悏慓慆悰恫怟悊恹怷忺忥徺怟怐怔彨忳忻弣忓恏怮恅急忼怐忩归忀徱怯怳徥忊恆怒忘怿忒怨怘忋忄忕怕忞忦怱怐恊怰忬怆恐忄徲忿怴忈必怎徺怃恇忿怹怙怭悗悕怢悕惓悖悁徲徢徽忇怬悗怄恡悋廮廜廼庯弍廳徝徨徕彉彑彥快彦忋徫忧循彙彟彤徸彬弝弗彵形怸忾徶怮徾忔怛态怄弆弮彾徝彑彣弯归彺怞忮徸怱恇怱恂徻怓恒恘惀怇悸恞忪忆彅念忢怂彵彝彄廽彖彪廽弻彣弨彚彝徕彝彜彀彐弄彣彞彃张弱復彳徯徔弮彽彟忶忌徉忋忌忞怔怖忹忁恅恃徐征强彨彲徨彲役彮快忋快怫怐怰忬恉怿徦忿徻当弟怀彨彍彽役忻徘彅彇彭彬弸彽役徇徽彰彐弱忐彋弲心復徂彪待忈徕彼彮影弰弹彿廦彗廩弔廬志怗忓怠忝怤忲忲徿忊彺弪徯徒彵徂当彈怙忰怲忂忰忕怐怶怺徲徉徝忰役弝忉徽怟怉徼忛徶忉後忈忬形怩怜怦怂怋恂恩恓恈彻忙忕彼忶忎徉忂忯忈怀忒徤忖徉徨徃忲怐忿怠怕忌忸怑忹忁忯怱忝怦忶忡徽忡徙怨忚怼怗怶怰忂怺忁忷忘怸怕怩忏怶怘怃彂弢忈彖微彤忍彔往弚後彉徤忔彖忔忣従怌懞录忸怔徢怷彏忦徆惟弎恜彰忐恘徼役张悟悡庠徃忪彠归庚徆愜忬彋廬彟弣庫弘幋悋彳帲忱彯庙庈弭廵慈怲庀徎忭弊廁年応怂怂怹思怴怱怴徻忚怘忈忉忋忝怼怆恉忛怃彆懔忆忞幜廚彖怞忥怊懩恂怄忉彚弩彳弭廑悺恲弓度御德廴怛廦愻愋彯帳怕徝庥怖开惐惌忤帓座弶廦徟庖悿恳徲巡弲彫忏怣忢忙忔応怑忛怠忯怠总応怮忨怆忿怿忖彋恰徏強憾忙弸恖彉恥怆徜忓慢徺往徟彺弦愊忟彀恝彄忂怌弝徜怞彤廴慢忕彁怪応弞惄悀庮扣彟廝忄彿廿恎怿床惜弗庖怐怖怱必忪怅怽怆必恇怸怬忖忂忚徵怛忒快悞弿循惏弔悵恻恂彼徳忟忪悝復恋忦悃怰悃怍恵恺愈惃恬怂影悙彄怕愔怸怛恶忸徶恬徖怈慵恫愪想忂怌恿彭忶惐惁惥悼忬怄忷怯忹忢忬忯徿忥恃忨思恃恀念忚怫忡怾悆彻彼怒忈彝律彍弴怟忖忊恃忽怼忲彳徴悀徿徃忪彉彁彶忍徥悗復徒忒彿徴彰彑役悪彎彥忂忉徍徂徎徝悤徶徽径彍後忖彰徠怮忉忁忓总念怆忣怨忨怞徸忨怇忄怇応徼彉扼抈悔廃拘彸愠愮很悝悷強弞悓庁徘忒忛愚慑徃忘憂徂戀憙弇患愹弇弁悥彬愴恳式惏愕廍康慆彌愌惱怸怭忚心忐忴忒怎志庮愹惞幻幻愤廽悕惿德忹怏忶怦忶怐怤怭慛庶廚愐悏恷憗待怽徕庰归忓弫彆悅彠弲愋愹怽扎怓忶愌憀恩愼廰序憯弯庾悎弩徼悵弝幽悾幃幤惪弌弼怿忳怉念徾忈忌徺忴惨廫幤恭弣廲悉弴彞恃忨忣応怕恊怨忳怎徜慽恃弙帾怈彇惤忡怢忌彉廴帨廗彬彺廀心忥怢快徕悁恇廐忓怔忕弫庁庆彑彘恹恀恗怇弉帾幈恲從忒怘恁怚徽怀徾怐徸怊徿徿徹弯帢嶪役廋彦弡忸怛怘徹忄怅怛怰忺徥恅彼库并忓忖恻忈弭役彺庆序庁廊径恁意怜怜御念怯恟惦彈忢徫弼廞徉忁忾恷悥徚徏彊席弽彜彙怰心怷徶怇忄怜忣怕忨怔归徴异帱廓徿忣怗怜忸忋忖恁徺忏怒怸忹庾弁彂忀彇怛录忉徏徤廧廁彉弣得徑徱弲恷徐彵思忓忟悙忇怖徥徵弹弛彖忊忖彯恋弻廡建忍弝律形従彵忋怊怭忘忈忪恃怹忆开廱廔徟弞弈待律彚忞忞忻忸怑怵怰徵忍忽徕庒怑徲幧御徉庢幟幟弐彌廧彟思彟怯弚恦弿忁彻廎徥彺廈忛悃惆恩性怃恸微惢弥忰忳彞彌忋恠形惄徥弩従彸彮徵废归性弃廌悐彪徍弬忎弘恒弿影恝廤弛得弄弰怿径彥怎徫彲弋彘弈悕怣怓忝忲忸忝忏忚怓恁恇急忈忔志怏恈忆恁悠悋怛忴悚応恡恵徶悞愕惨恐愲怮悪惐忕忂徧怞忮忠徺徬忳怏惘悂总彩悃惿惖慏怳悩惣悦忽悶悳愇愔忕怃惈忲弧悷恼恋愨恀忿怷怞怉忴忈怑怑忒忧忯忻怗忽忉怂忤忺悖惋愺慆惼憅慩愭忭忯怣怂怮恴急恓悃悠怶忾恿悤怘惽恆恚徦往悄悝悝愻怐恂愦怇悯惤愡愐慹憃憜懕忥徰恑恁惘悻悃惑愤応忦忑怂怗怟徻忪忇恄忒忰怘忙恁怾怅忧徺弋弻弘庩廱弥庖廘彽廚弤彞彁彧彗廋徎忌彶忤忧彯往彨忟怯弭弟弉彇弅弲徕弑弱征廿徕彋廇往徤廨弿彚度归徐库弔弪弄廐応徵怈忊忩怴忶德怓忖德快忝怟恋快忎怨忤徤徎徱忓弢彝彏彏忁徂形彴彮忐忘徵彘恖恙惱徸忣惸徂忹律徤忷得彔很徍彭弧弧応徍徃彞徾彾彚徿廧徊彡彬彫径彝徝彐廵忚忨忚怸忩怴怉怀念恈忐志忭怌怕忶思怮憶憍憂戝戹慽懯懫懥悃忤忽悆怋恰悯悋恔怊忾恛怱彜怕忼忷惡恂忕徹忬忋忨忯忬忰忷必忽恧怄怚恢恐恹忧忘忨恃怵怯忇思恁怔徟忮恸徿恭恽怒悹徼忻怈怓怌忇徵恊怏悒愳惈怱恚恟悆恥悗怂怅徍復忥忕忴忾惖当彬彖徆弅弅彝徖徑忞怃忸开彌徨彩律恓彷彼徠彲弣形彃弉忤忛总念怡怢恈怯怊怯彫徵彵彑徎彳形引影志怜怫忲忽怞怃怇忸恩怿愄怙忯忦徙廥徤徉徊怄彪徦徚弞彑微引彃彐廡廕廨彩徬延徨很弮徰彶应弶弜廒忤怅後忝很徛弻彗徨怮忭怮恉怪徾怖忚忮怈彼彔彵徉弉弝廧彅忡怽忸怵徶忬恉怲怇总恑怰彈恏恷忳廕復徖徦往弟彍弽忞弄徼悎忷得悐彶录悼応忸役徫徱怖徭忀怸徘彤徺徑忘彳忤徆忚徎忡怊恇怖忴忂忹快怕忁忦徲徑彭忍徙彰張徾忚忸态忮总恂忎怂忳徉彘徇廰径廫弴彭徟役彊忏從怓徜彋弫徱怑怳徸性徵怨怪忁心徜彺彵异弊廨弈忢彵徱彎徊彈廮彵彩弶徙忪态徸忮忩怟忨忼忚徑徕彦徥弐彦彷弽徺怞忑忻忞忓忇忊恈徼忄忝忎徸怒得徱怅怛忋心彜循彅徿徕彁徾忘徼徚忒徎循徛心徜愻影念徹愔忑弳征忮怤怊廙憈憵徔廲愕惝弭忘徖式忳弜彞并惞归忲徭愁憿徽庈忺怱彼惟忓感抍徑廝恠愼徱怖弫惭扰徏廃悒感徽忼怟怃怔恇怠徶怍忸徸怍恇忇忰怽怶急庫帧思惕慑徔彲忭怒总徥怚愋慇愕弻廥廒慧慡悠惪怼恝復微强惛惵徨惨憛慜必忦彭惾情恁慼愱恦弎弙弭愅惺忔憮憺慛弯彑弔怢忺忪忹怣怢怴怗怆忦忎怒徶怿恀忮忒怈徝庛恡怀態怅弳徽惑忂库慻忌憠忨惌悞彸彧徥憖忛幈彎怖怟彴怎幟懖愑慈悙怼悳恪忧彨憤悿恓态怯彭忔忺彸憠愭憾恕恼愨恦怠忩忖忻忽徹忇忬恉怏忚急怘忋怶忌怨怼忒忍忥忏径恭弢廰惔怹徥恙弇悕愁弖怔悷恌徱弁怯怚弣怌怤慭忷怑弜恑弗忮床幷慏從従徣彰彅怼廗廧愌徿德态忊彁恺廡庡悫恈怃心怗怊忢忯忀志念怣忂徿忯徵忁忩恁憻悏悏悼悔徽恠忪忯愞惚恜怵怜态徢怕怆悁忨忐悟忕忲悄忋志慾情悃惞忒怳恕恐彳愸惵恽怿恇怞忶恶徂慾惉恲惤怒忶忒怪彀忇忏忤怢忪忷応忦忨忡怚急忔徹怘忔忁忛怆怨彯廪御恉庳弣廃慰恐恩庉悡怺恅彐恕彀彋徒惢愤慒忂忦徙彬忾愰念惡慎忭廵念悶忇悠忬惢惚悍庛廲怺怿怺忙忈怔怣怠怳恱恬息恣慜愝徼幒彟怯忤怞忢忖怅怗忽忣庆恠廹当恟彏恲彇庶怔廳廸惣径悵徫式惧弰弾彵廒忓循弳幏怑怟急幦忑怦惂怰弆感恍恺廇悁恛愓悦废忿怟忖必怇恃忑怴忹忥恋恮廀怼恏愓忶弇悛忑応怓忪忟忶怤忔怆愗恗怉忳彝忰怘怶弛憃廗怓慥怴彾微怈徦応怜幑弥廞恚惙悃彫循彖廫恥惼忨忔恁帶惮床幂悫怢怓恁彠廉怏怒怺忘忾怓怫怂恄惵得徬愄惘役徿惵廓忽徹怙怑恀怍怣思忱弾慫徂徳廏徲廨彅忆彯愾弑怟忦怓悽廵徭弅弰忏徨徧怓忂弤彊弶恻徶怙恍怌悊役徑彀怮忶忾恙彺慄弰怂怿徸怫忮忋怰忮怽怯廩慟徢徘怇怣愥彖恃忻徽怼怓怚怱忌怬必彝愱恑徘愦徿廰悒慛忥恱当忸快徬廲忘悙忸忲忒怷後恨怲徺恒怎愈恼彶惗徿思彾愳怼恺怸怑惖徰弤怆慡怦忀忽徸忞怼忇忧快怮惐怗徥悀徤弩恉慠忡怀忤忱怦忝忆怇怳廊徣忔弖徑廵忓怽悑弒悕忽弭愤憟幽幵心弻忈従徥库慃怌廕忰悆惠愕惀悒愪惛惙惧弳徭恱弌忸总彠忠怸悐恹息悉怠急徼怫思忈恣惁弡怄怳德恈忺徆恊悡弡怜恈廡徽怌恋悶悤忉怸惖徧忚悛忻怊忟忨怆忥忺忹忣忶怍性応怗恇忹怕恂恆怫怴恱悅悰恮恍恰怌恲愅恖恺惵徵恀愌恞忣徦怶忋怓怫忭恝恶恫愃恄怲愬怦恐惜怫惄惘恙悾惽怛惔惬忒恆惘徹恈惄忉恇惖急応怮忣怽恂怄恈恁怎怸忤怤徼忍怾忖怚悳恧悶惏惒愣愵愍慺御徵怂徘怍悅忇忀悝悲悭怽悲悽恷情悲悘徆忒悹怺怔悻忀悐悪忪悉悍惧悥慕惘憎憆忠忶惝忏恆愮怬息惜怚恊怉怮忣怓徿徹怺怕怸忍怱忀怜恃恀态徎廾弜強庳形弛弲庱弖弩彾延弆廜庳庡廎彛忕彪忁忋彵怽怪怫廿弟开廌廒弛弎廝庿廮廬彎廐広廵延彤弸庳廘弝廰幀庹廉庿庈怹忾恇怾恄德怋怠怮忻怄怠忒恈忐怮忕怄徉弪徘弶徜徕彠彂忙弹弤建弝归弪廪弱弽怍恞悢恝怸怿徘怗惕弢弊彈彊彟廨弙弞弳廏廆廣廰廝建彖弌弒廬廪库强廡庶彌弒廭怦怵忨怠怍徻忤怘怐怿怲念怜怛必快态忞戰慡憙懇懰慍憭憌惑惇恼恓惔恮息悁恈応往思徽忇徱怏弭怶恵応念徖忊忻忈忀徵循忮忁忶恾怓怬怣怶忣忓恈忻忯怢恅忇徺徻必徫徬怸忶怼忤恝忾忙怸怨性忆态忓怾怞恓悭忷惇愝悾徢息恗忕忄徿忙怉忙彻怠惠廢幺庝广庍廿彍庵弗彻徛弦彦心怙彈徻当彳录廙徰彫徸庬後廛忷徸怤怅忂怞思怢忲彗弶廡徂彪徑异弽徕怕忱怉忺忓忸怠怺怲情恷恹恔怇恆弻忎彾恠忲徛応彄彻彎廫弿廖廭彨廈弌廾彵廋廆徴弰弔弚弔彠弘彅廠徘彰录徼弾彠弾徜弻徻怭怊怜怊恄怲怜徵徥弾廾彙彊廳弡弮弁忈忸怡怠忘忟恂忽怉忤忉忢彛恉徹廌弢従忯忍彷彈從徐彫徠彶弋徎彫彆弻廽弖待廂徖引弦徙徔弯徣徒彉徸录弣徔式弤弸弅張怽怤忙忪忺恄怛怭忦後彘廯彰弳彋彨弧弔怩徺恄怜怀怱态怙怡徠彩弹弟徬御彾徭忶忋怃忉弭忝忘忊彮徏徿总快怖忤徒忣徾忼德徕弦彂弞徹彡彛徳忘彛归後彸徬忎彷徔忔必怾怛徽念心怠怫徛従循彦彑忇徎忰微怦忨怫恂怚忙忔怯怲忾怷怇怞思怸忎怕恌弿忦彄彤忏忴彘忌忺彞忝徨忡徼彉忥态得彪愛忡悾徦怞忓态懔徙惛忇廘彂恂恫庴扦廮悦弉怼廾徇幇底弯廌懅弞忔康怃弔廊怑廑惜忆徾庹怎廃庛忥廬惑彯忱廪彰廅庣恋忑忝怓怑怦怐怷怹忯忘忭忸怛忢怰忐怶怳悎慢徚彥志徠征忢弓怀悌徕彍恽怟忮弿怈彌恀廞忴彯弥怺恿忾悌慩彴悦忺彔庾忷很怸愾恪強怄徛庬徏徢快惿徣彼彝徱庞徎忍怯徼恊忊恃怃怛怴忁怃忞怃怶忏忥怳忑忦彌愘怶忻忰憜悮懑戂当恍忌弾彃悐弇忏惀彟憍悏悉愑愬庮憅愘廢惣彗彛徣愉庲愕憢庪惿悵恂愃扟怽扯拋廽悖怹怭德憩彸慽戯怭忴忬怈怐忔忣快怄怤德怌忾怩怫忁怑忽恱廦弬彔底应彃庰异彧彏弑徧弥彂怂弒廇忣彩怞弧彺忧忣廆恳恹廚彈彰廦弙忒弴弥彡幻律形廀徸忎廥弒恽庑弶弒庙弮忂廖廌恁怇怙忷忯态忎徼忠忚怕怘忰怆恄心恉忁庶弳忎徯弳彁庮怀忮徲徘徣怓彎径怣忉彺彫応応怄徬忳彣彔徿従徸忽忄彎弬徝忭彔彁従忔徼廄彄彠忋强廿徢徽忬弍往彚忯彺忽怹忰怮怕徵怰怴怱忺忱忟徿怹怀怢忹恆拀抺徤廹彳挆拆抛惺惮惭徤彁徦惬恤愇恉怠慐循彇德恚惒愴彭恜慫弽彔徟怲惏愣弬悈惁徉弻弫愔愑感怎德忽忿怏心恃忹怫忺悤愔幼廦廻惐惧惮徙恅怌忳忙忤怹快徶忉後弹忌怛怆庩徚後惪忠弇彋悃彺怵弯彣志弅幽弖怏怲慅怉恩忹徳徏当彀开忎弁影怩弐巑廪弡巻彏弦弭強怤怍怯忧忟恊怭怲怜庴床庡弖幾弍廣弢徳忞恁忑忇恋忣怹忑怆恐怤彮庯愿廋忚廡恺怲恖弍幮忹廫彦库循弦徺弢征彺忒弉廧愈忷弓廵幙忂店弚彛悱忆怇府庴怞弸廱彣恎怡快徻忷忪恉徼怸忳怉待幝师徤异弆幨怜怭怮怞忱忬忈忘忐忱惁悦復悝恖悻憣徖怛必忹弥徵恱徜彫怆彆徦復徜恩恀恄恴怈怯悜怴徑忴怽恗忥怡徖悙忕弆忎徵彾怱彬忤怷忮怺怳怅忌怾怼忨怃忡廏徢忷忽怭徐忞怕忱忱恄怎忱忔恄怘徝彥念廻弞忛弦忑忑彛彼徻廵弜徾徇彆徛怀徆彾忦徠忦恊忳忏怞廽忉廰弴彸弧彊徬忔庩応彧弇弗徇待彿快怕怆徾忚志怨恂怢徨弗彅廷弫彅彔彉弼忞忠总怍怕忖德忽怱怜待忶徜徂徻怄怋志彻庹序悮弱彯彊幍形彂彻弅忤忿恐彣悲弔悾惐悕怀愍忩悴忣忧従忓悫往忬徑徐徎归惠恳徇徦恊彁彳強怑怒恞怂归恮形彛弗徠徻恃徾彯怵彝廯廵復惆惡忍循悻弧徴彘彰怗忏忒忪忀恆忭忌忇忯恇忚忯忻怼态怕忾惭悿患怑怠悻怌怮徛悲想愞恃悚慙悍性恲怺怨怨怫忰急彪怄当惟惀惓復悤愞恳忷怦悃惌愲微悗悾怉徤怊恅恨惶弧恐惎忉徲徬忀忢怴怯忖怓怈恈忕忥怢怜怎恄忉忴怼忉怪惐惹惣惦愗惁惦慵彗怿恵彣忟怼彬怪恏恱悕怴惴悅恌恀恝恁徻悕恗怲恡悩忆悁慆恄悸惒悑慀惿慗慉慼徬恲悼怿恠悇忳恪愛恄徺怯怊怬忻必忈忕忒快怳恆忥忋忟忶怛径彪弅彖弥彔弫序弜彗彋弔彰弁弤復彫徦忔応怒忷徏怊徆恖忆彙徣彗徨彜录弮彟彏徙弹待弐张彰彻弖彜彤弴彂当弬弖归弩弊怩怖怡必徸怭怲恊怕怙忡忢恅忯怶忋怦忁彽廵怎彲彞彘徔彳弲彙弄彑弎弶彍彫忠彗怙恖恍彔恢怶忍彦忋彻徺徥彪徸彁彗徬庸徂彂徔彜彼引忆彌弎徘役忉弨徧張强彙庽忩怒怽怿怈徽忴怸徿怒怨怩怉怞怱徺忽怙懢慕憅懆慏惯戹憕恁怡忣恜悆忯态惇恆忶怸德怀怀恜彾徧忛彲怬徸怉怯怪徾怞忟忎忷怦怠悸悆恠惓怍徱忭心忘怱忍恊急忨态怸忣悪悅惄怺悩怩怂怽忥忊怣忲怘忢忡忥悊忁忂悱恣弴怬愯彨彶弪徇徰怤忔归悙忞怙必廤弎弭怗徛怸怦往弤彛彄彪弆彎怉彯幹庤彙彶忑庫弡忂引忒徼忑恃忆怨怮忓恋微弽弫徉彌庼後怋役徻怫怴怿忑怤忹怺忐惠從徣恉彜恐庪張悊徸彎弹忋庼张弡弚徤开弤彽廜彀弁弉彗弢徻庵弡徭弶弈彞弾彰徺幍彺怴廨廞彊復怱忈忎忦怀怰怅忩忣怪忶庑弃彵弬弓建彂徐怡恆怈徾忞怩怑恆怄恷待怍彛恢思忞归従待怌彯式忘彴彇弿怂忱徨径忾徘徙徫徶徇忎忯忙応徵恂忌忤怈忔徤彪弔忹律徠徰忊恉忸忻徺忭怭徿忹怍徵彫彘徶忡忌徊当怚恈恀忍忰怵忤怽怤怨彚忱弰御徧弫弻快徜弑徿念徤徢彰彏徸弰徍彁怑忍忱徠恄怍忨徉弚徍彭彈彐彡忤徠弆徿徏忝忼廾忊忔彑忷忐忄恈怜怋思徿怟弤徒彨徏忶形必徔彆总忊怸徾德忌怅忇怒必忿怗徼徜怏怮忪忠徦弒廢徇忎弽怢快忇必彌很念徵徯徫忉忉愗愈憪惍愨憂慙愪慔彻忰悊彈忾恋彟徧惉惞悮恱惐悟恊怩忐怗恅怭惎忏怸惌怡恄怬徇怓惞徙怊恧弰忀怗怩惸慂怃悘慘怓怖悒怈怽徾恅怆怿忙怽忊怞徻忓怪忭忓怶怕恃怬怼悶怰怦悲怤悙悊怵悞愹忕悖慈怤怩惽応忐忘恟怴恎恏恈怂忻恨慂恡悉愰怴悌憉恅惕惙忨恙態恑惺愄徻恙惩徧怣惦忬恡愔忞怈忣怖怠忹忋怢徻忣怏忐怇怼怣忢快忽恶患愎慊惹慷憓愪憆弴彑恧徔心恖彭怭惙愚悼恜愮惵惷愦愧悢徊怓悌怊恂慑怎恰慐恛悓悼惆意憄慵憋房忖怈愯忞恣慠恱悖慆怿忧忈忆怎忭急忣怮必志怠忪怨必忟忋忀归廖庴廅弑庳彟弈庶廾弐彂弜弻彔弄廰弥忇忟徳怩怞徠怭悄恽廗庾弄库廁廨廈庁弙弲弓延弇庾弁弲庣廼幦应弗幒幨弈庼庁廣恉徾怙必忛快忴怖怐徺忰怪忰忎忲怅怛恈廔庠彲弝廣彝庻廅彃廠府廈庽序廄廌廑废怳怬恀总悇慛怮恣惎庆庢幔弄廌廐廑廑庨帬庹幷幥年庚廅幽常幘席幜幙幥帹幚庍帞忌怵志怦忐忸忾忍怟忿忊忛怉怯忠恁怦怘懤懃憌懻戋憼扗成懻息恫怚惄恡恃惥悊怠忱恪恬怏怤悓怅忖恘怢怤忱怮怐忡怀往徽恀忹恸悙怭悁恉悏悤忢恃徿忸徻忋怀怈忻悆恀悀恜恬悂恧悉怣恀怺忟念忯徶怦忭急悷悡惲惞愻慣恞怛愞恉怪恊怅恘徽悅怙恳帍帽廃帒帧庡廻席式必徺恓彧弾忄弟彥忩廋廧彦廩庳彄店幈录忆忺怘怨忘恊徶恈忼彙弫徍彟廩得廅庣彻徶念怄怇怓怸徿怼怀惾悗惨怪怇悂怏怓念很徲徤彿弞弨弴庱弝庙廜弡幫庙庹廊庳弘彃彔待廃庂廰廸庲彄彺彉徚弫彊彌弣彞微恂忼怆徼怫怣忑怒怯後徐徃廷康弤廌廿彁忺忟思怞恅总徿怪徺怟怹恈徒徳怃徫忙忛彫従徴影廳弻彐彐彺彪弙廹庴庭廗度廔幻徘彋彟弲廤弭彔彔弋弛式彊廌库庢录弒彇怬忹忮志德徽忳快恄彁廸廎弽弾弤彥廲弇怘德忷忶忥恁忡忥怒彏彅彏彂彶廧徶後彆彙徫徇彭徘往忎廱弬忄律徶忽忁忓怔怀怖徵弹弱彤异开彳影弥彵归彖彵庵弶征弹庻忾怾怗忸怃忟怓忉怊徢弾弐御弫廠徜彛弡忝怴忮徶忲忦怮怃怜恕怠悞怘悕悆怊悂怭怈忦怡徔怑徾忲忤忨忋忩必快怡忇怊徥忰'''

conv1_bias_shape = (8,)
conv1_bias = '''忐微忽弛快彰忓息'''

conv2_weight_shape = (16, 8, 3, 3)
conv2_weight = '''忊循彐悎彇怭志怚恔忄弚怨愄怅悜忎忛悯忱徘彥徵怐恕弓忹徨快弻慒总廙慛怚悞弳彼廧廱影恇忕彆徖怂悪恒徑惵忼恘彶彊怙惡待悻彉德志异悩廀惛彏悶弗徨廠弃悓恟悟惣恈忇悓彎怡徖惩慅慽惰憣惆惵怞慓惘徽弧彅悀悅恧弍悉征嵪幨庳帯床復幇庚幵彲彪惨恳惧彡悉徭徏怉徤忈悷怕忀忁慏悺悟悿忏怃忋徫怊忹彐恧愖恷忙悵悒惝愜忯恆恛廮悅忤徣彥悅彂忇德弬徲形恍悶彿弤彛悤彉廹悩怵廸廁忋彂悪彰忙态徇従怾彈徊彁弲悆彚悈怓怡恨彿役徵弌弉弭悃忬恈彏忺恛忭恧忴彎忹怎恭异异怛弅弾徤弽恰悷彂弁恱怊彉惒忥忏徦忁恸恙性忈恺很憓惉恁彧彄彊惀恊怎恄弳幂徴弄嶧御庠巋巴悼恝悤弦患徟弪徠惝忸惸弩徴忲思惺愮恸弄恿心恇弣彚恳怳惔弱彏彡恗怕弻恘惷悚徯怮怬惔悱快悅录怢恘惉彲慵忻怪憿怜恞态徏忏忙忞恼悲彳恸廧帢巵帱归工庌彂帉彩悳廽弔怊悀归恋彃愃彇彏悶彏愫悸怒彸忀忌彌恎悦徢怨弼忉怗弨彤志恥悅弞悄忘怺愋悬怠彘忣恬彲忉慺愍懸惲愌慭悹愝愥徿怡怕徖忬恍德悵悬幙幻微应嵺师庭崖廣忳惎徭忔怔彴愅怊惡慒忌您惀惔慷怀憏慨彷悁忪忊悘恓恲恶忢惆怸悍徔悲愛惵恮悂忝怿弁廬悖徽廉悄廢悉惗惻愛悼愬愈愰愒恰徥恀忡従忝怮悘怚幬帝弎彄彂後庄庑彞弳弩悂徺彴怦怳徭怤彍彼悍恫悹忴彔强彩彜忥悖忌怗悅怏悝引怱忖弔徵弒恿復忉忦徔応徒彐忈形忖彴恒慈惁慿愭恛愩惾惯恽忦怘惻忕彘彔御徴思帥廐帆左帇平嵗嶼巠悏弲性弬恶怾忠張惰怦怸忾惌徫怫惱忽忳弒忓惐徿徰怬悷彨悝忼惗彺恑惛德怯悻彩念怯彔怩廹怗彼弶彋憁懐愤患懭愲悦愄怡徱徃悟徤恦弬恠悭怘嶍帷幺帆嵃廌巃巻师弅悔悵弚怲惗弤怅恲忺愅忎恠怍怺愣悒惸弱忩忲微恹怱従悋惓惍悄悅徍惄惦惕彣悔彟怲怢惼恋忂忻徑彜愷憁恬慉憶憀慚愠悑彍怑忁怮忄彵怐怈恳巠幰廨庵嵁彠帑嶘嶉怼悌彵忛忶怊悶忘怛憃怱愧恅惽恏忺恅怰彔恿快悖忰恋悇恢惿徵惂怏忸愄慞愮悙愉怹弛怸恂怋庨忘录彳悼彘徔怶律怯慀弽循廵彖恕怘庉恧彤悈忙慎怣徿庱愫恮当怛悕彨恪弒庖徖怺忞徺庾庐往徆廼彅廊恀忇徥幛弚怔彯恰徸悲徬恜幚廔性廇怭恙廦恖徭恼忼愅忌怱恵惆惙悝愒憦恭愬惸恏慞恁憫徫弯恸怘悷惺忦惈從巀嶞幡嵋幀帽庠嶦廹忦怠悆徸惔惷念彫悞悕惁恠慉悫怭怽徹恌彼徘忦怶惯怾彋彌徏忢彯徔愍彭徤忦愴悽怚恍怳徨悫忺恻徤必悧慝慨愉懞懌悅恐愨患恆忞彙徿惊惣恵惡幝巻帡嵘嵦弁巈嶥帋悧彎惏惢恧彰忌怮忳必慵忿忿愐慦恼慶惹悰惛怱徭徝悛徶忨愖微恸恢恃徦悙怠愘恡态悼怍怛弖恹忱彬徬微恍悅弣従怨弶怸徵恋悕恾恐弶恴役忸恈怴忟彸徧怕待怗忍恮彃彠彉情恰弴您惍忪彸忐心恡忂悔悈御悥彅忯怅怹恻怘弔徛怡悝恺彁弞怚御彉徕弭弛徎忙廛廀忑庙庤庥悝憅悆復惺恡愒悃慟徝彼徖廕廨忪念徾待廨微彛徂忨廛彷帥怓廎忮忨怆怶康微庣彗恹强怼庎廪忟忁弢徦开廬恏徧恄怱怡怜怦径彗徽彷恤廬悐徱弓悊恞徂從愀惂悜忞從慦悬悤憃怩慿憌恒憗怹徬总徝悟惆惀忋悺嵫巰庁嵣嶦己峫嶲嶖悌徎恀忌惿愇徹忠归愮悴愁怃憃恲悦憎憣恊悄怉徢惦惌徜彵恌律愮忴忿恭悈怶惖怀'''

conv2_bias_shape = (16,)
conv2_bias = '''弎悗彝态恛愼悢愬悩憢廻愗惼廳愾慵'''

conv3_weight_shape = (16, 16, 3, 3)
conv3_weight = '''恹彙忇悪忕念忆忡忤悌怲恨恛得徆彃得彠恂恁怗径忐忞忨悄快径彠志彪彏怕徖怌恮微彐忠恰忠恺怺恷恖急怍恼彭復忤恘徤态彖恫彔性恬恦恕忉悉悎忩怖怮悖恂影彅徒徫恧彥忛彞彺徇德忬彯悏彞恫往志恳彇怦徖従怬忟忈急彸恔悟徔忑恔従恡怃怍悂怏怪忨彵徻怵悔弼恄彋怨忡恋悘悋恞忨忍恈恐彼徽怎忶恀彨恟怺徊怽彩恑怔彃恈彇徏忔張彵忯弩怚忱忰徰怎怯怜徾徐彦彅忽彘怟循忤徧忻怞彮忤徶徑彘悆彝恾怇德待恽徦彘弢怍恡弻徔恭彫忪忸忝忒徳忶恺悊怞徧怖彁弮彟強志弴息徭役彮怐恜恍彏徇恘怒彬恀彺恵徇恹忨忲徙恝彏忙忛恦恴怼恹怳惼彗愝忳忠德怨徣恊忐忥忿彸息徶怟必快徍怠恕恆恟忮恤徳彿忮悙恷徱彰忒悝怔弶忣弞忙忴恊弾忟彼息怓弶忓忈恡悊徏恕恍恁恐悡怨恥恏応怍彯怠徣悉彝悅彟悗悅悞彜恁恟忸彭恪怛恞弼恑恦忴徒律彥悗怠思忬恳徭思彜彸彤怼律忊徧怦彻徟忯徥彣恔徸怍忕彙徝徑悡忝怘彬恗恽忹忏恦徢忟徻归怬恗彂忷忀悌徯怂徾弻徶恱忣忠怅恌恉徕忳忴怓恵徹徬忯怾彊悉徃恎恀悔恉忖恅恋彀恖恠性怒徘徽悂怑徇忡徹恍彣悓恄彺忎怎德忾忙思忑怇忖恋恏忢得忀彾復彛従徼徢恴怩忞弽恜從忬怋总恾怟恻悬怈悆怔忱彅彗從彇徶急恺忎徟忍恪恍徍彖怖彙怽徦徯徏徸悦怉彰徶怑徳忼恌惀忀惪态悻徫徻忻往惃悭总徼徧恇悲惗惎恄忱惃徬徔徤恤恰徢忴怤怅恞恆応惨悛怱总恾怕恫惤徶徸恔忄恴彏恌息恕忠恲怼悊惃惊悪徟惖怙恛惝恅恌恝忨恑徖徙恌悶徨心总徖惑徜恍悒悮怴忐悆悦惭悢忿怘悲恦悯怺悛悀怃怿德怔御循悈恮恨恩彥後弿徍怀怶弪怊弴徚恭循彞恮彪忥得悪彤彶彼律徫徘弜忪弮弸怒忞忍恶息悀彞彙得忖彃忺強弫忠心恞往微彰徣恥彤录忴怶忘忠怾恵怭彘怠弩忠怬徹得恵忍彝弻怚归恉忾徃怭录彆弧徸忠恨忸恨彿德徣恠彫悍忕怢恧彸忟彑弼徜忴恐忆恨忘恄恉忀彸弨悪怆徇悞恖恚怕忖忘徿忌復徨恊循恘忐怽恘怆彃恳微忉恘忐徼忊彀弳忽忇彘忶彬怸怰悞忭惸态怵恏恵徻忇徏忁忿徭彐恽忯彶徶怏恨恟怏悤忄悚怉恠恅忙忬恅忘德悻律恻愐愣悵惉恁恇愠惂徴悙恂悗惨怲忭悆徺悭悛息惇惦息忢忧恋恽悝恷悝怠意惨恾悚愽恼惓惞怅愭愗悗忱彵恄彔徸怴恧徔悏怱恤悬忙惷惣恚惥怊忞悝愀悳惊惒悭恚恙怊忯怇恕恮恚忓悲彶恜愝怲怙怱怔悸恸怱悯悒恻怀恒悳恟悄恟悭忋思彽従悄徽忥恷彩忪彦怢怃徎忩恮彭恭彦恦徺忔恍征彗彧忟怇恼彟怀徤忲彝彵彩悖心徭徜彘徛忼德影恅微彵徔忙恟恚忶彞恠徜彼恍忔恡恑忪悞徢怊彵彣忒徏悍恲待徺恚悜忖恷彻悍恴忖忭忾忘恺徫怤悚忍忋怏怊恪彛怭忂忿怡恲忒念悎恽忚忨很彍徦復忂徱忆恩恪彛彐恛徇徲徑忹志徾恖徔悔徺忝恰怖径忉彨径恊忿悁怞彮怂怾恃恛恋循忧徛恡恸怊忐徜徘恝怵急彩恮恧恙忆彶彭怟怤怛怍恌忪彼忹怾怄恗影彜弹彆征恠徎忒必徿怵徰念当得怃強恠怟彡怷彈恻徑忏怌恫悇恲徆彳弸恤急怅從怟彧忚徎恽役怭悄徛忸恎循忯彠忠归徊忡恑彇恝怙忯怚彷忰恗恴恙徶彅恫忐恗忦快忘怫怈役徉怆怙忏怂怓恇御恷待恜彾恞忶怑彞恁忓念怤忐急忡性必怈怕忚役彛怏彰怱徎恶彌徙恠怹徂從彬忾彷忄御後徳忉彄徕德循忹彏忯彁恎忺彿悆恓恨彂怘彚徿忍录彳怈彵怖彶彚恐忑徣怯彴恵恋恅徖必徊忷怯彨彚彥怍恱応徯弹彆徊怗怮怖徑怱忋忽志徊怬忕忏彡怊彗怡彅弽忯恊徺怡怰恊忧徚恤恖忧徍当德彣徧怇怦彫怑忾怡忧德忐很彃徵怋忋悈彼後彡復徵恌怙忒役怀悆恃怾恭徔徍彽徏彮怾悈急徟彭忯徖快恵忧恽忔志忸忲忶弉往廔彔形忦弆恊弫怍恍怓復彯応彩怊恚徰怮徧徤恨徊悀恒彎彔恶徥彌徣彌怋徔彚忟怫弶怳彞徜恖忄彞急徉悞怌悒征忺彚恙彁徴怟徬恺微彼怽怄忪悒当怭役忓彬忴忭彵影役恸彲御悅恤忌徰悦悂忷怡忆怂恢悫悃悐彃性惓恠慦悋彐彍徱忾彦怰彮恁恿恞徃志恒恘恇彊忻彖怓忣徫忆恭忑怖恗律忓徊彯怚忈徤怂彠恬悞彩悚悆怚役恱忠彬忴悕恾忴彜怠恸忪怼從怵怚恐怈恥微影恝弰恍忱恩恩恎彦忇怈恓徭徳徴悐徶忕徖彟影徂怒恠徻彼彪徤彼怐徍怹怛恍必彦怽忊忎忲怎忟恘忆張悥怘怆徉徽強忼弳徒恢弬怸弴徖忆恘忍彣徕徫很怂徖徕悊彗归怹彾恁徍徤忄悛恢彶怘恏徧徫弨彔怷怀彻忉忕徾徳彺彅恳恔徏徥彴怵恧徻恬悂往怆怽怅恆彚怩忘怇怒彚徍忓徶徆彇恩忯弹弿怈徥怘徊怢恋後弯弯忕很彏怈惈怠悺惙惛怒徲悐復從征怍悦忝怩忈怊彩悥悉悂怍悯彠悥恗悧忠恗忽恏恦徧徳悁恛忈恸愊愃怪怖恘愒怼悊悪恦忰恮悠徻恤恲惨悑悇忄惪惥恖忰怣忼恖恾念怛怪怽忆悥恾怀惬悻惪惵悡想徠悂恋恿徨忛惇恻悫忍忯恻惵惆忤忡怎忘愁惴怿恜怛忑恬怰恴徟恦怌徍怂恷怞徸御您惨悢怅愤恂忲怱悌愕忯悱怅恏忧悤惔惺悑徢怹怟彲徻恢徐怾怘惲恖快怛悥悈徣恌恣徎徱怷患恢徛後恖恈恼彦忟怦彫待怉忻恻彯悅徢恛怢恺忌悳怤悜悥怤恐情惵怈惄情忝怟徢恁惀忈忧惗怃怈恐恹徾怴惀怀悿悳忠惖惜恮悬悻悵忊惜悂忭惋惶悄悕惑悊恒念徟忱怗徖忥惽悺情想恱惢忖怔悒悋悁悾怆惿怶忱悼怳怤悢微役怬征怳忣後徙总忱惇想怷惻惚愉悑惣悊恄悻恅惩惽怰惙恨弫恈徨彨恔彗待彑怛惄怨愹愞愖愁恘悭惆忹恶恛怨惌惁怊恁忆往悐忬悬怒惂徣彦忺悞恊徭徹恔怠忴悪惔恇愻愀恜愕慜惬恧怙恂怌愈悸忟悏怇忾怈悩怪怯愢恱惯愶惧慉恩怬恂惖怦悏悂怀悂慻慔恐愺愛愕愎悕愴忏恨怒恝怂怸徛忍彵怍愯愶怡怛恕愣愲怚忺愤愈您悜愑愼恉惺恳恼忾徶彲悙怖怽御愥慡惱憴慊悥恼惌愤恺惒怾愽惌愝怘慏惎怉怆悢恂徨忥恩怆徘恜恒恚徔彀恁怤恋徹忇恦快忈忪復忕徛徵彠悗恒恢怭悂徇怗徕彫徿彬徘従徢從徝徢怞忮彉忛忚怘徚徠忤律忻彆忆恶忡忸彪悃彔恸彲彊忹復忴悇怟徫彠念恳怎忚息彛忀悁徟恲怈恊态得悃忰忾怤忍怟忊怎徫忪彮徏忾恪忨怠怋張彦恝忓归恟彉彣恤悆忞恲徺恚徙恰彳徃恱悝徚悆影忍怒怊怛待忎徲怨恊彜徆彡恈彋怟忩忋待悐思彳忌悧彰忛彐忻徥形怐恸恍徚彚息悅後忾恞後恔忌忰恻忤忇忇怴忘怂恫忬忧徖恮怃徥徒怚恔忨怦忢忢弲得悚忋恬徻徶怜張彭怄怩志急忛怽悖恼彑徚恶恱彪恕忓忒悲恹忉徟徻忆怅徕怔怨彽悏忐弽怹徚恒忋彦恜恿彭忹悔悽彀恼恄怏徭忷忒怢怬忕忮忭恷怆忙恳恊徙後恦恊徱彞彣忒徙恪怅忞恲怾恼徔怐怀徘弼忣悇恋彾徝彈忲彬'''

conv3_bias_shape = (16,)
conv3_bias = '''徇怡彌愄強愫录怫怦怵徍怿恥愍徚忋'''

fc1_weight_shape = (64, 16)
fc1_weight = '''徊恗帣弪庘幂惖慪惵慩廼帜憚慲悪徿憅恎愷徟慞慩悛廱开庮庙懒戢愱愂帰怾惺怨惿慭忯形幦愨慟悧慉懎廕庾戃廞憉愂悾幹彿怯忯徴廸応慧庞忡恩愐弡彡总慇懏惔庽徫戉戢廈户徸承惈戮彳忱弐庢悢徾廃懐憣徔怦怣弄恘恇怠惸徯幵憁慃幋帕悠庫慯恣彈帷愗慪巾懂庵愷带懔廞憯恚憧幫怸惖庥怘愮情恊悦帴廃廹彁忙悵怭庮廐庝彠惔彑懥恍惱很惰懄彡戏庭徟彅廝恵悪戌廥思懌慟怜憨愪巢惬慹憇彅巵庍憨怙怠怒憥憊彗幊庶彿慷意懟憲幟必待廿廂帏懃恉悃憕憑惓弆忖復怡廑怦廋廼帟徔忌您徑庆悓幔悯懚怹愍忈幄忨愽怇憑憌师弎忓彌弴帔彦年帙弖怸帰庉怴幹弼怣憃彾庴彚幜帄幰慗懽慽愠废恋弎弬慞庴恩忠慹愻幊愮影引彵慹慍悬憓悍廬懠慀慉忎德愩彖帑彋懇幵弗悢慿幸帎悽怟恀惜康庺悓恞愭悃慤店懡幍慇廇怏恬彝弡庇廦庰愙开彝帐愹憝弲慞恭怷徵幧徕惞庿彄庪庼忰徸愅廧怨帏弞忣廌忤惙弉廒慫懗廹彙幦幻延怌愥憴慃悋式彪怹惱帮徒彝惃廒悕感帨彂恶懒恥惕弻懖恙怸彦懄愚彿彚弶彵彮怏幝徬忦彁彌应彗惸庭懘慷應幼惯怀惐彴愅懏悮憘怍忕愡恏悛悷悚廃幺怲恕御愿愗弴彄憺怏恩惸徧庹悕帡彂幘庍幽帴彫恧懚懆憕恒庉徢彔愱帥帄弿忡廹忢廌彆怛廆愷度憨嶮愞怳悁懕悁忭廂愂徤忨惸憆弭广愾扔悴懩憦憻弸戓弿征庻惻快愝廾怎慇憠戰恫庚恡帟情彡抟忧愒愖恽彅忞懽愄慙抅戣徼彌异幏幁悰弹憶憦慊怏愲庞弨慬庮慑悇快惒廝徎廼憬幺庻徯懟悵慐徝忴忷彶巊懺恌慕広干慌徑慼悑帧庠怯愮弭悜憳忨恁怯廌庾庨张幊惗慍弾悻庐怤憽忦幉弱愐帕庤息忁弇怱愧愿廲徫惯惧快幽帄強悈廍帳怫庢忸快慽廀庘懩悀忱慆彻忻忞愆悢恟廓憬徹幔憢弝库待怛愻恃憋憮徥忉慾怑帱希帎廨帞悴惴忊悊悌悰忯慏惤悴憁憤彴悘怟慩庺彻弋庼慁怈慨恲恅幁御惻憐憣慉愧帴悚徫徺思恉恩悤懌御慝怂幙憖感徹帊愜忽徯征幻愇懼忌廿恇愮弆悦帨忘忈徶惷憺憏彗广彜帳廌悳您帬恎怯廗弽彊憈恶怟惵幖幁待惷慸幁怞悡庿廼怞往弼憚惕悚憒彧幪廗恚廳徐悖恙廏怿慢帙弍怢懏怅建怑惒帰悙恧慛恠懎怹幙府弔戃师幗彙忒弋徝憥怴怿帽情彷彧惟徰庙慄弌徼息幬忳憤庤府惏憻开怾忸庤庹慉悌徺懩愒憺幠帴徝庹廓廜帐廡廛弙彊强弘感惡忦憪惛幇弨彺懴恬慃徹憂庒愽憼慓従應愲徂彾払庎忻庁怚怿惪廤廹惊彈忍影恖愒必廨帓愠怉怪式恰弽戽慜悇慪忒席忹憗掗愕恑弁徚懼愱恛愂愋怏幇憽弛悎庈巻慨廓帼帊彬忆庉怆忢愢廃彊廖忱忔悬帛忀忠懓彯懇彑懦廳悄愱恝急悽慫惛慗弸懸忧幠憇慌戼庳彂成懨恳慱懌彼律愬惻慆懐床悶并憍帀彴弧慙幺惧当庤慘慉慶慶廊彺恩憍帰彐径憘忀廐怽帉憗忧弬恓恪悊愙巾弽憁憦慍慽憈憎怼幟弱巁惵嶱幨恢币幻恃嶵懿忶惽怛慒庇忬懂弓徨慝彻庑愙彄幮惚悉廕愢愗帾'''

fc1_bias_shape = (64,)
fc1_bias = '''弎抹愧忥挺悛悸憥庀懶怹庖幥恵廻廍戡幯慜忧廃弒幁徢懇忙徣帚恾扺懦扅惰帹慌幊怽弫廈悑嶳扎懟庆怰恠懲库愔往弴惪悛帪捰帯帿忓扛循庙懌幦扃'''

fc2_weight_shape = (50, 64)
fc2_weight = '''怇恴惞恅想愠悑急忂惨徎怑徫愤忖恊惖彬忥恁惲悠怱従愘恥悷恨恷忛惣憁形悧怜惤徘彅恷忣悡慧徹徳愭惏徤怀恆恕徉悛忴恠怞徺徘慢悢弅彊怗徜恑怬忓怊录悴恁弩彯彥悪得徝忞弈惊徵彲异忕徺忡恻悃彐徘恙念悢怀急徿惽徇悾徸徉怭惫惜惶忮惚徻悥征怵徼弍怞很径張忍思恁忯建惔彏惌徣徣廳心応延徔怇応怕怕恍忇怵彙徂彙悥悏弍张恈後忭弟忆忭悱幔态延怫彸往廭归恲忸悬怗惤怋悌廸怲庵怺徰循庺忳恰怿恲彅彖幯惂忋待忘恜彿恗惊恓恺庉惤恽忝張弈廸恀惈弋忚廰悕弣彂徒悺忋忾従徝恌徾弐惞徼悉怩悢怑弋怪彰悤忲弳微怨弰怼悥惃悥彾惙怸恦徟忶彖悘弭彐徎悁徹待悓從弨怅恹息废悦彡徺恤彂忾怦弚徇恁忀恼想彏总惹役悅弜忛恩忻徳思惏恥忲得悯徟弛弦彝彭惓怊恺怷弔忧強惋弢彡怮恵弶徐忡得彴惉怚怰恳怆怺恔思廮恷弒廼彀恼愉恏忲忾悺强怟悼惲態彡悅彣徲恕很恢弁恅慀怦态惂忡愩悶彫怈忣応悃悈弆忲弽恩悯彊弼怹弓径愸惛惥弧彧悩彛彺彄廨愎彷怊怯復愖慓怒恕恔役悏恦恱徢弢惱恆恣弸惧很恃循徚恏徏忁怢弮忟徿忔怢悩弌恏徦徆廝復怖忡徒忓悃恏律忉惟怊彚彑怼忟弹悼悎忶怴惡惇怩徥惝弫弴怀徝怜忶悃弊惋弨慌恫康徿庘怒弦恻徴惄庾廚悓弱怖想廵徕惴弃忆微彭徿彁恌彗彘悟怳徫庈弊恺弌怈怢廷弒徻弅悠库忭怽忖庨徸怚您彞悵怷彽弚弟弳干恘恁惀徸序彜帱庣彯怉庭度弭廮彅必徿彤弱徇恚徝恋忱律德彯悩惨徍惦庯御徹怀德彝引弩悾惕恐弳弹恡惗怚悎恉廌怗彉彝彵建彼弑惑惄征态恵怮徱弇彾怇式廀庪怏惋悍忯彀廐待徺弇悊思恍怾感惦惆忔怩悰徏恢怬徵惶恵悞彷忪恵忆廾弗形彣忦彞徰彘悪忌忒忳怉徎彉惃忈彂徏彺悵录彘彙怸忝忩弭彐忈恆弃強彫怠怯恫悆恺悱忂徲徤忞愋悭彿徱弾归弳忣彶怲忧徊忭怜息愭忳忦彏愛徑愁怟彉悸彏忺弰怗悻愽忦悛忯彆徝待惀恬恼弦彊徥意弁徱弫径恹惺悴怍惬恗惩徚廭廏弇忢忂怼怿忚怅忣惝悲弸弉忇恚恨彣忊恣弚忸弿彋忨廾悺怭恒徹忕弟忆悀恰恊彌徛悸徴急徥悚徸怛彄恝怗弐怖怅悕您怮徱恿悀彫怶弸悓怵弋彚恺廅弃徜徰弙式弶忲彩当徊彂弞延怡惊忓弩强弆彅恽弈忳惉律性悉弻廄忱弋患弭张恸忹忙怡恘恨恝怺徕待恎忢恜徧廗徘悥忮悢彬弫廠彴忿怭弘恰忡廲怋弅弓忶庳彗忰彋性忎惥徆悾忠弪怗彛忐徚惎悚彭悧怷徎怜悍弉忿恞彥悲忲庒悅徝悌徱徻恻廊惫恒怂徐廇廟忭惽必弝彧彾很怰廾弪惠忓廾恃徛恌恮廬恋徣悑徃怤恰徃悍彘彏悃悽彩忣怺弧廕恡恕彧悋庶怳恈弿徘徎心怂德弿徖心彍徻彆悒徔悎怫忺悫怆愈態弌弢彉惦悹弻役弐恃強徲惱惑忷恿忸徕怄徲徚情徫忺志弇憲惩忁快忨怡彈惾弍徿悅悛悍悕悻怍徲慣廬忧恌惛惇怎忩惡怑彐彔忑微庸惋忎惈恠慆忛徽忕惮弄弫慝惘憀悠愆恻惨怷徛慤恾怠忑彖弑弚悰悊廤悻忰悪徵惐廵悏恞怭忇悅悂恹忉恛悺彿惙彌悡恱恇悪彞恥怐怨弁庒德怕徸悮忋惓悾悩廴恸庳怀廱惈彊徳惂惹性彯廷忪弇悭悥徖彥恛忽怃彸徧愦廯延忮惊怛怞怕徯徫影恗悃弭怂愊惐徰弐徒恸快徧悰忀恕徻怹怈悕恪彡彔忽恞悪忀忐惒悡徣忂弯愕徭弇徚徲悙彖怴悾恎怪恔惪徐恍必徃怭徂怟廁忧往彛惃怰恘恳惖廋徢怞怶忞弯忤慍惆悌弹徼廜弚彅彍忝恹怖幠怃悁復悤悚恽志惀弻庱忊恰急彧帽怄弜悃怨恓廞忎弗怴徃徱徣彚弍恡徸悏忙恌弃弻德惚徜影怶怈忾彴彀弐惣怸悎怷彇巊悇忥忛彙庿弭怸忞恶廚忄怜悚恊忄恺恒弔徙恚恌恕廊弭悭廊快恒弩怓惥怣彎彂廎忇恥徏忞惿悱忺弣彌惥搌擘恓斴怿懟徵復掓慨惡彟恉忪彗摱悀摗悸惻忯慀彳斪搸惪忣弪晙搣擐愙徯忚恤忥微惃惠弈昷搭徥悩忓昿捝惙徿怴愵敗忰橍彌捽搸授惕恓搂彁挕恰懊悤惸惧彾忢忖弻慸录式彶怹徔後愑彳戦戦悞彋愎彙憽怖怂彔悠恓恅愡怹弟彶廙忳怂悠惢怼慀憝心惾悴惡愿怳忱弯悞慶徎扎彯扆惻怦忝恉愄彘慀惱悧悟愢懎悹惈怎悎悛悧忒悆彪彰惤恻徺愋恗徿忴愺怉懀御彟怆惸悇憞懡怛忌径待恢惠恭惜廴悀慿悠怴徢恟惢忹彧患徻感悹愑悋拽忍恦彸悾恫忄憚恝惚愹憔愩彐惮怇悟憀恆弸惽愴弃开惊忯憇惲慄弥慬彡怴悶忱怔惓憇忲惏愩弾忩怩悗御怷悿庋悛怺徛念愂従惙愉弊彸彄恕弩扗彩懭惉恫徙恆忣忽戂徦岒嶞役崚廛弦廡徢崵怦悃开御徛惷嶣恁嵃徰彻惴悳惚岿嶓怼惛応媱崗岊庛弖廷愝弁忮弊恂必宽嵣恚恑悼娉屡徫恿彽彼寃恛嘊弜巂嶤嶀恴彧宱惸巉忦徺愙彀惸忀惙彽悑惿忛惏惘惀徘怌忾怕恓怜恴怴彈徺愲恔怜廱彲徰恗怖强悕彧忞惱弉忥弧患悥恰快復彇憚徼怟悞弋惇惷惥彰恎惥悫循徖弿忧悴愍彜悞愈很忳惵恳惈想怈怆強彞忭怔弸恬忬愝恎恾怙惰怽悔愀怭式彺惪忟愡彫怨怗徆弾惮忐愾惞愛愞忱悧悩慓悃徳彾惤怣徝徨忀怽彲律慧恒惛慁徍悺悭弪彂悒彡很悦彝彑怉忷您怽恞彌惘弨恖徊徳怛怆忟弦弥悬復怸怕彜徍忯恠惢怸悌彀忭忤怩弣悏弻忢彇怊庮庨徟徤怛恆愃弈恙怜愕彀怲循徰忬怛态彾恚怫悒弋恊惝廸彲怙忌惺忚怊恖悎恛恟徚恺弇弡彗徘廛惐徦悒弈恧怍弾恓悼怪忂惇弻弻怔徂弩弪惽徧庲忚悲征弲惆廓強德怑恧悍悲庽惽异悎怭徾徝彾彝恛忤怬弶弚怟彉弉怃怣徝弹必徖悀徨惇悰忋徥恜怋廅悀忒德弈彁廼忮惶惕悩悸德徴录悗怺弛弡强彞悘彞悌忥忆悛怡徉怢忊彾恊徨录弆怘弘廂彔愠忘忔思弘彰悮必忮恽後恣惑悋恻忶惰忀引忘弪徧彣悯愙役怪彁怊徵恅怤念忟悔怇弉惻彿怱忕惥忱惃徥忑愌悁弋彼悍恏彅忴怰弮忪従怓弅惗患慆悁愮待彳惷彭廻怦恼徰悋彠悾恶心怈悎弥恑怦怚忙弭弞愍徝恤怰弾彞惖悎惚彭徠忁息怵恕悵恲恔愊弑彻恄怸廰悏悏怙忆忼恽徍悈徃恧惇恜弎彘恮惌征忤怆廾廆惆怟德悈忇忖彍忄恓怔忮徽悁彩徯徬彗廾恃悖忛彔征徜彤徸弅徻恬廏廇怼恜悢悃弢悍忖怂忠怮帪彾怾怙徴徖悌彉归忚徭忂廳徿忡廠待彞总庎彅彞恤徼弴弸彆廘復怵怉弰忚徎徽弴弻恔惋恷悲怗徶怦徛弒恻庽彗彃徯忞怗廨忞形忽恥廤心怭悭彈悠後恜廤怃式忎忹怎彇恨彉忸廱忹彩彁悆悍您惀怫徼惑忔德忟彺怇徉惁惗彐惹愄恮徴忉忈悾彏循彧忸恌徭徭惝彉徇怤异惆弙悷徬徫彛惆怋彴徵忪悼悒弔徘忻怃怏悫彡惭彘往徑怐忼恵悄悗庺忼忉廫忋悽恛急恓悋悐惆弻恺徐悏张怌弐忧忶弼恉悶恝悅彐怉悖徣徶弱恮忓恦惨恸庐弸惂惃怈悌忋怣忨忡悑恭悺弡形忺弤彌彭恁影徊息徹怋悃志忛怖恏徃惍悍怋彮忽徰忷忐惕彥悢怑悳怉惭悞弗徽徚忿彋忀式弮徚彔恄忆恂惭志忈惄式徆彔弉怌悊惈惇彣息弶怈忌徯彄心怃怾徿怖彚怓悽応怎恊彫悄恾彅待庳彯怯廰恁彯惨広怭彩弝恨恴廽忤徘弃徴徙彧往忓弡弘影徃怂弄忰彠弶悌悆恿悦怦彥徙广彚庸徛弱怳徬悡悚忒弁恔忽恅忚恉惍彞怰思恣徰忖弶忑弒徫恂弉恫惈徑惙忧徢徲恺恲徯弐悵惸忖彵弛恒徭彣恁弓忮弍恟怎忓忻忐怋徍庖弦忿弲彬恮弩弚彜悊悰恕怔悐彑志彰徱彭彧忐悱径廜徻怺彴彮彪徎恈恑您愬患惁恀怸忀怪弸徥忉惫悌悿忳恹恣弯彄徕彐慖悐忱徉徫悎怒愩悃徹徶愥径怃彞忔悒応愠恨彂忥恿彼征弸征恩恨惘悍忞恴悈愁彜悕怱怗恕彬彦廅庬幊徦廖悑悧徧廹彋怳忣彼彡怢怙徉彬彀惁忽惧幤復悐恨忟忊弙忔悟怗强徙恴徵彅悞怵幪弉怚弄彊巖愤彃忡弭悐彠徒怼忺徂徨廐忣忧庎忧廜怯彞忔惝悴愕恂忍弫怣恇悅恷彵強惜弧徕怖徳悘志强徤忈恷悟徺得惣很悕怙忔惶悾彵悔恀彜店御徦怒彬忣徘忼惯惴忍愕悪悠您徆怃悟徻彼忦弃恹恨弜忩庒怺应恗徹廤彷廻恉惯彣悥循忳弟彺廀恌忰悩怈徾怱恫恣悕弩怢志廨忞想廵忹思彴彖廴怸幃怃弦彎弆弶惌怊怏弪怌彠微徻忬彥幷庵徒惻忚忽徾患幫怯彎弊忨忒悘恛廕徍弯怖彘忤怱廍彷幞忺怲怖忉德從庳惝念彀徜彖徬忕弆恵徙彾怳忎忬忘弘彈总庶怋弉怆悊惻恜恥徇息徥徫廴怌忻彟彷忠恇帘恶怒惃恀懗忣怔忠悔慪愿彟恭恒式彃懸張患急徴忈徵弶惴恝惧徐愷憴态悁惝弲怆忤忷征恟彶怠慾悗怗懬慠憼庚恶恋弪後惓怇彩怗彨慛愺怈彸怳怇憧廲彅幅庬嶱廵廯怹悦嶬彼惰悙恘微怟徤弌忄徬恈悔康弖徂弥恽惠悑庒巶影恡彧恰恮弿待张徽幹帼彻徏平幵性恈彄徒惏恤征惽左悪怘帽庞彇惐彨悞彼怭忴幋忰弖惚徝彔怚往庣恫愣弓彜恝庘彭弦恏悙彜忊怌廸庹廚弻廩弦廻弹微恠徭彡恆怛恜徭怢幻库弴忦徶彛廜征怗悳患徢惨幧怤悬忳心志彜幗悅彃弑帲庎廀庠惃庱恁惀嵚幖恋愚彶悎弦廂徭府悚怅恓待悘巒庿彨忆悩巎弉幾弨惴悧庭彛怕忨恾彄庛嶐恡庉恁幓徝悌彽悌忱弔惔巯彷悂式巓惦徊嶀彣彁彅州廑廒廈惙廘忚恣帉干恥惕悆恹弔彏恦弥忈怛忡怨恵嶧庑徆惕彐川幗巬徍彐怮廰廾恩怰悹常幐庆忍廂徖帆悇廤怯忱彸廿悇帲徫怄弢広彴悊嶷恰幱怤惈憐憜执忪憅怄惱愬慩忮弌张弫忴戊忽悀弙念恨怇弜憹慈悌德愴惎慃戶悞情怄徍徨惎忶恔忰憹懛徶慴恍惇悳弮徃彌悋懟怳扁怤彂憂慭恛惭憐徰慭'''

fc2_bias_shape = (50,)
fc2_bias = '''彻慐幛彏彏悠慏弆开彁徤悏庖彝弰憜忯悈御忿散憆慺息岸愖慟怯悝徴彼忋忈徯応循弥怙徊憊徺彎徴忄慶廞彼弛從愜'''




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

        self.fc1 = Linear(in_features=16, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_players * num_actions)

    def relu(self, x):
        return np.maximum(0, x)

    def eval(self):
        pass

    def forward(self, x):
        x = self.relu(self.conv1.forward(x))
        x = self.relu(self.conv2.forward(x))
        x = self.relu(self.conv3.forward(x))

        x = adaptive_avg_pool2d_numpy(x, output_size=1)  # shape: (B, 16, 1, 1)
        x = x.reshape(x.shape[0], -1)  # shape: (B, 16)

        x = self.relu(self.fc1.forward(x))  # shape: (B, 64)
        x = self.fc2.forward(x)             # shape: (B, num_players * num_actions)

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

		#print("Actions pr dites par joueur :", actions_list)

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
				if actions_list[i+5] == 4:continue
				origin = Coord(p.coord.x, p.coord.y)
				mv = origin.add(directions[actions_list[i+5]])
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