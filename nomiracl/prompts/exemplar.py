import csv
import os

class Exemplar:
    """
    """

    def __init__(self, max_exemplars):
        self.max_exemplars = max_exemplars
    
    def get_exemplars(self):
        """
        """
        return None
    

class OrcaDPOExemplars(Exemplar):
    def __init__(self, max_exemplars=1):
        super().__init__(max_exemplars)
        self.exemplars = {}
        self.languages = []
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        with open(os.path.join(dir_path, "exemplars", "orca_dpo_exemplars.tsv"), 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            exemplars = list(reader)

            # Row 0 contains the split and languages in the format "Language (ISO code)"
            for langauge_row in exemplars[0][1:]:
                language = langauge_row.split("(")[1].replace(")", "").lower()
                self.languages.append(language)
                self.exemplars[language] = {} 
            
            # Rest of the rows contain the exemplars (Starting with English as the first language,
            # and then the rest of the languages in the same order as the header row)
            for row in exemplars[1:]:
                key = row[0].lower()
                for i, response in enumerate(row[1:]):
                    self.exemplars[self.languages[i]][key] = response
    
    def get_exemplars(self, language):
        return self.exemplars[language]


class FewShotExemplars(Exemplar):
    def __init__(self, max_exemplars=2):
        super().__init__(max_exemplars)
        self.exemplars = {
            "en": [{
                "query": "Which is the smallest planet?",
                "positive": {
                    "title": "Planet", 
                    "text": ("The smallest known planet is PSR B1257+12A, one of the first extrasolar planets discovered, "
                             "which was found in 1992 in orbit around a pulsar. Its mass is roughly half that of the planet Mercury. "
                             "The smallest known planet orbiting a main-sequence star other than the Sun is Kepler-37b, with a mass "
                             "(and radius) slightly higher than that of the Moon.")
                },
                "negative": {
                    "title": "Outline of Mars",
                    "text": ("Mars - fourth planet from the Sun and the second-smallest planet in the Solar System, after Mercury. "
                    "Named after the Roman god of war, it is often referred to as the \"Red Planet\" because the iron oxide prevalent "
                    "on its surface gives it a reddish appearance. Mars is a terrestrial planet with a thin atmosphere, having surface "
                    "features reminiscent both of the impact craters of the Moon and the valleys, deserts, and polar ice caps of Earth.")
                }},
                {
                "query": "What is the primary language spoken in Laos?",
                "positive": {
                    "title": "Lao people",
                    "text": ("The boundaries of Lao dialects also extend into the North-East of Thailand, known as Isan, but the Lao spoken "
                             "in Thailand as a whole can be differentiated by adoption of much Thai vocabulary and code-switching. The language "
                             "is not taught or used in schools, government, and most media outlets. Thaification policies removed the alphabet and "
                             "now the language is written in the Thai alphabet, if at all, and the name changed to Isan to sever the political connection "
                             "with Laos. Despite this, the Lao language is spoken by 20 million people, almost a third of the population of Thailand, and is "
                             "the primary language of 88\% of Isan households. It continues to serve as an important regional language and a badge of Isan "
                             "(hence Lao) identity, but it is experiencing a decline in the advance of Thai")
                },
                "negative": {
                    "title": "Isan",
                    "text": ("The main language is Isan, one of the Southwestern Tai languages closely related to Lao. Currently written with the Thai alphabet "
                             "(instead of the slightly different Lao alphabet), Isan belongs to the Chiang Saeng and Lao-Phutai language groups, which along with "
                             "Thai are members of the Tai languages of the Kra-Dai language family. Central Thai is also spoken by almost everyone and is the language "
                             "used in education but native in Nakhon Ratchasima Province only. Khmer, the language of Cambodia, is widely spoken in areas along the Cambodian "
                             "border: Buriram, Surin, and Sisaket. The \"Lao Isan\" people are aware of their Lao ethnic origin, but Isan has been incorporated as a territory "
                             "into the modern Thai state through over one hundred years of administrative and bureaucratic reforms, educational policy, and government media. "
                             "Despite this, since the election of Thaksin Shinawatra as prime minister in the 2001 Thai general election, the \"Lao Isan\" identity has reemerged,"
                             " and the \"Lao Isan\" are now the main ethnolinguistic group involved in the pro-Thaksin \"Red Shirt movement\" of the United Front for Democracy Against"
                             " Dictatorship. Several Thai prime ministers have come from the region.")
                }
                }
            ],
            "hi": [{
                "query": "कौन-सी गैस भोपाल गैस दुर्घटना में रिसी थी?",
                "positive": {
                    "title": "भोपाल गैस काण्ड",
                    "text": ("भारत के मध्य प्रदेश राज्य के भोपाल शहर में 3 दिसम्बर सन् 1984 को एक भयानक औद्योगिक दुर्घटना हुई। इसे भोपाल गैस कांड, या भोपाल गैस त्रासदी के नाम से जाना जाता है। "
                             "भोपाल स्थित यूनियन कार्बाइड नामक कंपनी के कारखाने से एक ज़हरीली गैस का रिसाव हुआ जिससे लगभग 15000 से अधिक लोगो की जान गई तथा बहुत सारे लोग अनेक तरह की "
                             "शारीरिक अपंगता से लेकर अंधेपन के भी शिकार हुए। भोपाल गैस काण्ड में मिथाइलआइसोसाइनाइट (MIC) नामक जहरीली गैस का रिसाव हुआ था। जिसका उपयोग कीटनाशक बनाने के "
                             "लिए किया जाता था। मरने वालों के अनुमान पर विभिन्न स्त्रोतों की अपनी-अपनी राय होने से इसमें भिन्नता मिलती है। फिर भी पहले अधिकारिक तौर पर मरने वालों की संख्या 2,259 थी। "
                             "मध्यप्रदेश की तत्कालीन सरकार ने 3,787 की गैस से मरने वालों के रूप में पुष्टि की थी। अन्य अनुमान बताते हैं कि 8000 लोगों की मौत तो दो सप्ताहों के अंदर हो गई थी और लगभग "
                             "अन्य 8000 लोग तो रिसी हुई गैस से फैली संबंधित बीमारियों से मारे गये थे। २००६ में सरकार द्वारा दाखिल एक शपथ पत्र में माना गया था कि रिसाव से करीब 558,125 सीधे तौर पर "
                             "प्रभावित हुए और आंशिक तौर पर प्रभावित होने की संख्या लगभग 38,478 थी। ३९०० तो बुरी तरह प्रभावित हुए एवं पूरी तरह अपंगता के शिकार हो गये।")
                    },
                "negative": {
                    "title": "2020 विशाखपट्नम गैस रिसाव",
                    "text": ("विशाखापत्तनम गैस रिसाव, जिसे विजाग गैस रिसाव भी कहा जाता है, 7 मई 2020 की रात को आन्ध्र प्रदेश विशाखापत्तनम के वेंकटपुरम गांव में एलजी पॉलिमर उद्योग में विषाक्त गैस के रिसाव "
                             "की एक दुर्घटना थी। इस दुर्घटना में, स्टायरीन (Styrene ) नामक यौगिक वाष्पीकृत होकर रिस गया और हवा में मिलते हुए आसपास के गाँवों में फैल गया। यह गैस सान्द्र रूप में होने पर मानव "
                             "के लिए घातक होती है।")
                }},
                {
                "query": "भारत के पहले कानून मंत्री कौन थे?",
                "positive": {
                    "title": "अनुच्छेद ३७०",
                    "text": ("संविधान निर्माता और भारत के पहले कानून मंत्री भीमराव आम्बेडकर अनुच्‍छेद 370 के धुर विरोधी थे। उन्‍होंने इसका मसौदा (ड्राफ्ट) तैयार करने से मना कर दिया था। आंबेडकर के मना करने के बाद शेख अब्‍दुल्‍ला "
                             "नेहरू के पास पहुंचे और नेहरू के निर्देश पर एन. गोपालस्‍वामी अयंगर ने मसौदा तैयार किया था।")
                },
                "negative": {
                    "title": "योगेन्द्र नाथ मंडल",
                    "text": ("योगेन्द्र नाथ मंडल (१९०४ -- 5 अक्टूबर, 1968) पाकिस्तान के प्रमुख जनकों में से एक थे जो वहाँ के पहले कानून मंत्री और श्रमिक के रूप में सेवा करने वाले विधायक थे। वे राष्ट्रमंडल और कश्मीर मामलों के दूसरे "
                             "मंत्री भी थे। एक भारतीय और बाद में पाकिस्तानी नेता जो पाकिस्तान में कानून और श्रम के पहले मंत्री थे। अनुसूचित जातियों (दलितों) के नेता के रूप में, योगेन्द्रनाथ ने मुस्लिम लीग के पाकिस्तान की मांग के साथ सहभगी "
                             "बन गये थे। उन्हें विश्वास था कि दलित जातियों को पाकिस्तान बनने का लाभ मिलेगा। वे पाकिस्तान के पहले कैबिनेट में शामिल हो गए थे। इसके कुछ ही वर्षों बाद पाकिस्तान के हिंदू-विरोधी पूर्वाग्रह का हवाला देते हुए वे कानून "
                             "और श्रम मंत्री से अपना इस्तीफा देने के बाद भारत लौट आये थे।")
                }}
            ],
            "zh": [{
                "query": "月球到地球的距离是多少？",
                "positive": {
                    "title": "月球距離",
                    "text": ("月球距離 (LD) 是天文學上從地球到月球的距離，從地球到月球的平均距離是384,401公里 (238,856英里)。因為月球在橢圓軌道上運動，實際的距離隨時都在變化著。")
                },
                "negative": {
                    "title": "地球",
                    "text": "太陽的直徑大約是月球的400倍，但它與地球的距離也是400倍遠，因此地球看到的月球和太阳大小几乎相同。这一原因正好使得兩天體的角直徑（或是立體角）吻合，因此地球能观测到日全食和日環食。"
                }},
                {
                "query": "彩虹有几种颜色？",
                "positive": {
                    "title": "彩虹",
                    "text": ("彩虹，又稱天弓、天虹、絳等，簡稱虹，是氣象中的一種光學現象，當太陽光照射到半空中的水滴，光線被折射及反射，在天空上形成拱形的七彩光譜，由外圈至内圈呈紅、橙、黃、綠、蓝、靛蓝、藍紫七种颜色（霓虹則相反）。"
                             "事實上彩虹有无数種顏色，比如，在紅色和橙色之間還有許多種細微差別的顏色，根據不同的文化背景被解讀爲3-9種不等，通常只用六七種顏色作為區別。國際LGBT聯盟的彩虹旗为六色：紅橙黃綠藍紫。紅橙黃綠藍靛紫的七色說，"
                             "就是在六色基礎上將紫色分出偏藍色的靛。傳統中國文化說的七色是：赤橙黃綠青藍紫，青色就是偏藍的綠色。要是把橙色也分爲偏紅、偏黃的兩種就是九色。三色說有：紅綠藍，就是光學三原色，所有顏色的光都是這三種顏色混合出來的，"
                             "和亚里士多德紅、綠、紫三色說，就是兩頭加中間。")
                },
                "negative": {
                    "title": "彩虹色",
                    "text": ("彩虹色或虹彩（），是一种结构色。彩虹色常见于肥皂泡、蝴蝶翅膀、贝壳等物体。 如果观测物体表面的角度改变，色彩也随之改变，这样一种光学现象就叫做虹彩现象，即彩虹色。")
                },
                }
            ]
        }
    
    def get_exemplars(self, language):
        return self.exemplars[language]