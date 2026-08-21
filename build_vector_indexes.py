"""
Builds processed corpus JSONL files and generates FAISS vector indexes.
"""
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data.tech_data import EN_TECH_PASSAGES, HI_TECH_PASSAGES, MR_TECH_PASSAGES, TECH_LONGDOCS
from data.movie_data import EN_MOVIE_PASSAGES, HI_MOVIE_PASSAGES, MR_MOVIE_PASSAGES, MOVIE_LONGDOCS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

ENGLISH_PASSAGES = [
    ("en_p_000001", "J. Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II.", 101, 1),
    ("en_p_000002", "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.", 101, 1),
    ("en_p_000003", "Hacker House Goa 2026 is a developer hackathon focused on voice-enabled multilingual Indic AI applications and low-latency RAG systems.", 102, 1),
    ("en_p_000004", "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by referencing an authoritative knowledge base outside of its training data sources.", 103, 1),
    ("en_p_000005", "FAISS is a library for efficient similarity search and clustering of dense vectors developed by Meta AI.", 104, 1),
    ("en_p_000006", "Goa is a state located on the southwestern coast of India within the Konkan region, famous for its beaches, architecture, and tropical climate.", 105, 1),
    ("en_p_000007", "Sarvam AI develops foundational artificial intelligence models and speech recognition tools specifically tailored for Indian languages.", 106, 1),
    ("en_p_000008", "Devanagari is an Indic script used for writing Hindi, Marathi, Nepali, and Sanskrit languages.", 107, 1),
    ("en_p_000009", "TextRank is a graph-based ranking algorithm used for keyword extraction and automatic text summarization.", 108, 1),
    ("en_p_000010", "Singular Value Decomposition (SVD) is a matrix factorization method that identifies principal components and factual energy in context synthesis.", 109, 1),
    ("en_p_000011", "The human heart has four chambers: the right atrium, left atrium, right ventricle, and left ventricle.", 110, 1),
    ("en_p_000012", "Cricket is the most popular sport in India, with the Indian Premier League (IPL) being one of the largest professional T20 cricket leagues globally.", 111, 1),
    ("en_p_000013", "The Olympic Games are a major international multi-sport event held every four years, featuring summer and winter sports competitions.", 112, 1),
    ("en_p_000014", "Indian cinema is the largest film industry in the world by annual film output, comprising Bollywood, Tollywood, and vibrant regional film industries.", 113, 1),
    ("en_p_000015", "The Academy Awards, commonly known as the Oscars, honor artistic and technical merit in the global film industry.", 114, 1),
    ("en_p_000016", "Social media platforms facilitate online community building, real-time news distribution, and global digital communication.", 115, 1),
    ("en_p_000017", "Digital India is a government initiative launched to transform India into a digitally empowered society and knowledge economy.", 116, 1),
    ("en_p_000018", "The Himalayas form the highest mountain range in the world, hosting peaks such as Mount Everest and acting as the source for major rivers like the Ganges.", 117, 1),
    ("en_p_000019", "The Western Ghats are a mountain range running parallel to India's western coast, recognized as a UNESCO World Heritage site and biodiversity hotspot.", 118, 1),
    ("en_p_000020", "Artificial Intelligence and Machine Learning enable computers to analyze complex datasets, perform natural language processing, and automate tasks.", 119, 1),
    ("en_p_000021", "Cloud computing provides on-demand access to computing resources including servers, storage, databases, and software over the internet.", 120, 1),
    ("en_p_000022", "Stock markets enable public companies to raise capital by issuing equity shares, allowing investors to trade ownership stakes.", 121, 1),
    ("en_p_000023", "The Reserve Bank of India (RBI) is India's central banking institution that formulates monetary policy and regulates the financial system.", 122, 1),
]

HINDI_PASSAGES = [
    ("hi_p_000001", "जे. रॉबर्ट ओपेनहाइमर एक अमेरिकी सैद्धांतिक भौतिक विज्ञानी और द्वितीय विश्व युद्ध के दौरान मैनहट्टन प्रोजेक्ट के निदेशक थे।", 201, 1),
    ("hi_p_000002", "मैनहट्टन प्रोजेक्ट द्वितीय विश्व युद्ध के दौरान पहला परमाणु हथियार बनाने वाली एक शोध परियोजना थी।", 201, 1),
    ("hi_p_000003", "हैकर हाउस गोवा 2026 भारतीय भाषाओं के लिए वॉइस-सक्षम एआई और लो-लेटेंसी RAG प्रणालियों पर केंद्रित एक डेवलपर्स सम्मेलन है।", 202, 1),
    ("hi_p_000004", "रिट्रीवल-ऑगमेंटेड जनरेशन (RAG) बाहरी ज्ञान भंडार का संदर्भ लेकर एआई मॉडल के उत्तरों की सटीकता बढ़ाती है।", 203, 1),
    ("hi_p_000005", "गोवा भारत के दक्षिण-पश्चिम तट पर स्थित एक सुंदर राज्य है जो अपने समुद्र तटों और संस्कृति के लिए प्रसिद्ध है।", 204, 1),
    ("hi_p_000006", "सर्वम एआई भारतीय भाषाओं के लिए विशेष रूप से डिज़ाइन की गई स्पीच रिकग्निशन और भाषा मॉडल बनाती है।", 205, 1),
    ("hi_p_000007", "देवनागरी एक भारतीय लिपि है जिसका उपयोग हिंदी, मराठी, नेपाली और संस्कृत लिखने के लिए किया जाता है।", 206, 1),
    ("hi_p_000008", "मानव हृदय के चार कक्ष होते हैं: दायां आलिंद, बायां आलिंद, दायां निलय और बायां निलय।", 207, 1),
    ("hi_p_000009", "क्रिकेट भारत का सबसे लोकप्रिय खेल है, और इंडियन प्रीमियर लीग (IPL) दुनिया की सबसे बड़ी टी-20 क्रिकेट प्रतियोगिताओं में से एक है।", 208, 1),
    ("hi_p_000010", "ओलंपिक खेल हर चार साल में आयोजित होने वाली सबसे बड़ी अंतरराष्ट्रीय खेल प्रतियोगिता है।", 209, 1),
    ("hi_p_000011", "भारतीय सिनेमा दुनिया में सबसे ज्यादा फिल्में बनाने वाला उद्योग है, जिसमें बॉलीवुड, टॉलीवुड और क्षेत्रीय सिनेमा शामिल हैं।", 210, 1),
    ("hi_p_000012", "ऑस्कर पुरस्कार (अकादमी पुरस्कार) सिनेमा की दुनिया में सबसे प्रतिष्ठित अंतरराष्ट्रीय पुरस्कार माना जाता है।", 211, 1),
    ("hi_p_000013", "सोशल मीडिया डिजिटल माध्यमों से लोगों को जोड़ने, विचार साझा करने और त्वरित समाचार फैलाने में मदद करता है।", 212, 1),
    ("hi_p_000014", "डिजिटल इंडिया कार्यक्रम का मुख्य उद्देश्य देश के नागरिकों को सभी सरकारी सेवाएं इलेक्ट्रॉनिक रूप से उपलब्ध कराना है।", 213, 1),
    ("hi_p_000015", "हिमालय विश्व की सबसे ऊंची पर्वत श्रृंखला है, जिससे गंगा, सिंधु और ब्रह्मपुत्र जैसी महान नदियां निकलती हैं।", 214, 1),
    ("hi_p_000016", "पश्चिमी घाट भारत के पश्चिमी तट के समानांतर स्थित पर्वत श्रृंखला है जो समृद्ध जैव विविधता के लिए यूनेस्को धरोहर स्थल है।", 215, 1),
    ("hi_p_000017", "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग कंप्यूटरों को डेटा विश्लेषण और स्वचालित निर्णय लेने की क्षमता प्रदान करते हैं।", 216, 1),
    ("hi_p_000018", "क्लाउड कंप्यूटिंग के जरिए कंपनियां इंटरनेट पर सर्वर, डेटाबेस और सॉफ्टवेयर सेवाओं का उपयोग आसानी से कर सकती हैं।", 217, 1),
    ("hi_p_000019", "शेयर बाजार कंपनियों को पूंजी जुटाने और निवेशकों को शेयरों की खरीद-बिक्री के अवसर प्रदान करता है।", 218, 1),
    ("hi_p_000020", "भारतीय रिजर्व बैंक (RBI) भारत का केंद्रीय बैंक है जो मौद्रिक नीति और बैंकिंग क्षेत्र का नियमन करता है।", 219, 1),
]

MARATHI_PASSAGES = [
    ("mr_p_000001", "जे. रॉबर्ट ओपनहायमर हे अमेरिकन सैद्धांतिक भौतिकशास्त्रज्ञ आणि मॅनहॅटन प्रकल्पाचे संचालक होते.", 301, 1),
    ("mr_p_000002", "मॅनहॅटन प्रकल्प हा दुसऱ्या महायुद्धादरम्यान पहिला अण्वस्त्र तयार करणारा संशोधन उपक्रम होता.", 301, 1),
    ("mr_p_000003", "हॅकर हाऊस गोवा 2026 हा भारतीय भाषांमधील व्हॉईस-सक्षम AI आणि RAG सिस्टीमवर आधारित डेव्हलपर कार्यक्रम आहे.", 302, 1),
    ("mr_p_000004", "गोवा हे भारताच्या नैऋत्य किनाऱ्यावर वसलेले एक निसर्गरम्य राज्य आहे जे त्याच्या किनाऱ्यांसाठी आणि संस्कृतीसाठी प्रसिद्ध आहे.", 303, 1),
    ("mr_p_000005", "सर्वम AI ही भारतीय भाषांसाठी खास व्हॉईस रिकग्निशन आणि AI मॉडेल्स विकसित करणारी संस्था आहे.", 304, 1),
    ("mr_p_000006", "मानवी हृदयाचे चार कप्पे असतात: उजवे आलिंद, डावे आलिंद, उजवे निलय आणि डावे निलय.", 305, 1),
    ("mr_p_000007", "क्रिकेट हा भारतातील अत्यंत लोकप्रिय खेळ असून आयपीएल (IPL) ही जगातली मोठी टी-२० लीग आहे.", 306, 1),
    ("mr_p_000008", "ऑलिम्पिक स्पर्धा दर चार वर्षांनी आयोजित केल्या जातात ज्यात जगभरातील खेळाडू विविध खेळात भाग घेतात.", 307, 1),
    ("mr_p_000009", "मराठी चित्रपटसृष्टी ही भारतीय सिनेमाची जन्मभूमी मानली जाते, दादासाहेब फाळके यांनी १९१३ मध्ये पहिला चित्रपट बनवला.", 308, 1),
    ("mr_p_000010", "ऑस्कर पुरस्कार हा जागतिक चित्रपट क्षेत्रातील सर्वोच्च सन्मान मानला जातो.", 309, 1),
    ("mr_p_000011", "सोशल मीडियामुळे संवाद साधणे आणि जगभरातील घडामोडींची माहिती पटकन मिळवणे सोपे झाले आहे.", 310, 1),
    ("mr_p_000012", "डिजिटल इंडिया मोहिमेमुळे सरकारी सुविधा नागरिकांना ऑनलाइन स्वरूपात सहज उपलब्ध होत आहेत.", 311, 1),
    ("mr_p_000013", "हिमालय ही जगातील सर्वात उंच पर्वत रांग असून ती गंगा व सिंधू सारख्या मोठ्या नद्यांचे उगमस्थान आहे.", 312, 1),
    ("mr_p_000014", "सह्याद्री (पश्चिम घाट) हा महाराष्ट्रातील व भारतातील महत्वाचा पर्वत असून तो जैवविविधतेसाठी प्रसिद्ध आहे.", 313, 1),
    ("mr_p_000015", "कृत्रिम बुद्धिमत्ता (AI) आणि मशीन लर्निंग तंत्रज्ञानामुळे संगणक स्वतःहून नवीन माहिती शिकू शकतात.", 314, 1),
    ("mr_p_000016", "क्लाउड कॉम्प्युटिंग इंटरनेटच्या मदतीने डेटा साठवणे आणि सॉफ्टवेअर वापरण्याची सुविधा देते.", 315, 1),
    ("mr_p_000017", "शेअर बाजार कंपन्यांना भांडवल उभे करण्यासाठी आणि गुंतवणूकदारांना शेअर्स खरेदी-विक्रीसाठी मदत करतो.", 316, 1),
    ("mr_p_000018", "भारतीय रिझर्व्ह बँक (RBI) ही भारताची मध्यवर्ती बँक असून ती देशाच्या चलन व्यवस्थेवर नियंत्रण ठेवते.", 317, 1),
]

LONGDOCS = {
    "en": [
        {"doc_id": "en_ld_01", "title": "Overview of RAG Architecture", "text": "Retrieval-Augmented Generation combines fast vector search with deterministic context synthesis. By indexing passage vectors into FAISS HNSW graphs, sub-10ms retrieval latency is achieved.", "source_lang": "en"},
        {"doc_id": "en_ld_02", "title": "Manhattan Project History", "text": "The Manhattan Project was led by the United States with the support of the United Kingdom and Canada. Physicist J. Robert Oppenheimer directed the Los Alamos Laboratory.", "source_lang": "en"},
        {"doc_id": "en_ld_03", "title": "Sports & Cricket in India", "text": "Cricket is a major passion in India, governed by the Board of Control for Cricket in India (BCCI). The Indian Premier League attracts top international players and millions of fans worldwide.", "source_lang": "en"},
        {"doc_id": "en_ld_04", "title": "Indian Cinema & Filming", "text": "Indian cinema includes Hindi, Tamil, Telugu, Malayalam, and Marathi film industries, producing thousands of films annually with rich musical storytelling and cultural impact.", "source_lang": "en"},
        {"doc_id": "en_ld_05", "title": "Digital Technology & Business Growth", "text": "India's technology and startup sector has expanded rapidly, supported by digital payments infrastructure such as UPI and a booming cloud computing market.", "source_lang": "en"}
    ],
    "hi": [
        {"doc_id": "hi_ld_01", "title": "RAG वास्तुकला परिचय", "text": "रिट्रीवल-ऑगमेंटेड जनरेशन तेज़ वेक्टर खोज और प्रासंगिक जानकारी को एक साथ जोड़ती है। यह एआई मॉडल को सटीक उत्तर देने में सक्षम बनाती है।", "source_lang": "hi"},
        {"doc_id": "hi_ld_02", "title": "भारतीय खेल और सिनेमा", "text": "भारत में क्रिकेट और सिनेमा दो सबसे बड़े मनोरंजन क्षेत्र हैं। भारतीय प्रीमियर लीग और बॉलीवुड ने वैश्विक स्तर पर देश की पहचान बढ़ाई है।", "source_lang": "hi"}
    ],
    "mr": [
        {"doc_id": "mr_ld_01", "title": "RAG सिस्टीम माहिती", "text": "RAG प्रणाली वेक्टर सर्चच्या मदतीने माहिती शोधते आणि अचूक उत्तरे तयार करते.", "source_lang": "mr"},
        {"doc_id": "mr_ld_02", "title": "महाराष्ट्र भूगोल आणि संस्कृती", "text": "महाराष्ट्र हे सह्याद्री पर्वत आणि अरबी समुद्राच्या किनाऱ्यावर वसलेले समृद्ध राज्य आहे. मराठी चित्रपट आणि नाट्य संस्कृती अत्यंत प्रसिद्ध आहे.", "source_lang": "mr"}
    ]
}


def generate_data():
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    passages_by_lang = {
        "en": ENGLISH_PASSAGES + EN_TECH_PASSAGES + EN_MOVIE_PASSAGES,
        "hi": HINDI_PASSAGES + HI_TECH_PASSAGES + HI_MOVIE_PASSAGES,
        "mr": MARATHI_PASSAGES + MR_TECH_PASSAGES + MR_MOVIE_PASSAGES,
    }
    
    for lang, items in passages_by_lang.items():
        corpus_path = config.PROCESSED_DATA_DIR / f"{lang}_corpus.jsonl"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for pid, text, qid, sel in items:
                record = {
                    "passage_id": pid,
                    "text": text,
                    "source_lang": lang,
                    "source_query_ids": [qid],
                    "is_selected": sel,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote %d corpus passages to %s", len(items), corpus_path)
        
        longdoc_path = config.PROCESSED_DATA_DIR / f"{lang}_longdocs.jsonl"
        combined_longdocs = LONGDOCS.get(lang, []) + TECH_LONGDOCS.get(lang, []) + MOVIE_LONGDOCS.get(lang, [])
        with open(longdoc_path, "w", encoding="utf-8") as f:
            for doc in combined_longdocs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info("Wrote %d longdocs to %s", len(combined_longdocs), longdoc_path)
        
    logger.info("Building FAISS HNSW indexes and centroids...")
    idx_mgr = get_index_manager()
    idx_mgr.build_all_indexes()
    logger.info("All FAISS indexes and centroids generated successfully!")

if __name__ == "__main__":
    generate_data()
