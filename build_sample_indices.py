"""
Builds processed corpus JSONL files and generates FAISS vector indexes.
"""
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from retrieval.index_faiss import get_index_manager

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
]

MARATHI_PASSAGES = [
    ("mr_p_000001", "जे. रॉबर्ट ओपनहायमर हे अमेरिकन सैद्धांतिक भौतिकशास्त्रज्ञ आणि मॅनहॅटन प्रकल्पाचे संचालक होते.", 301, 1),
    ("mr_p_000002", "मॅनहॅटन प्रकल्प हा दुसऱ्या महायुद्धादरम्यान पहिला अण्वस्त्र तयार करणारा संशोधन उपक्रम होता.", 301, 1),
    ("mr_p_000003", "हॅकर हाऊस गोवा 2026 हा भारतीय भाषांमधील व्हॉईस-सक्षम AI आणि RAG सिस्टीमवर आधारित डेव्हलपर कार्यक्रम आहे.", 302, 1),
    ("mr_p_000004", "गोवा हे भारताच्या नैऋत्य किनाऱ्यावर वसलेले एक निसर्गरम्य राज्य आहे जे त्याच्या किनाऱ्यांसाठी आणि संस्कृतीसाठी प्रसिद्ध आहे.", 303, 1),
    ("mr_p_000005", "सर्वम AI ही भारतीय भाषांसाठी खास व्हॉईस रिकग्निशन आणि AI मॉडेल्स विकसित करणारी संस्था आहे.", 304, 1),
    ("mr_p_000006", "मानवी हृदयाचे चार कप्पे असतात: उजवे आलिंद, डावे आलिंद, उजवे निलय आणि डावे निलय.", 305, 1),
]

LONGDOCS = {
    "en": [
        {"doc_id": "en_ld_01", "title": "Overview of RAG Architecture", "text": "Retrieval-Augmented Generation combines fast vector search with deterministic context synthesis. By indexing passage vectors into FAISS HNSW graphs, sub-10ms retrieval latency is achieved.", "source_lang": "en"},
        {"doc_id": "en_ld_02", "title": "Manhattan Project History", "text": "The Manhattan Project was led by the United States with the support of the United Kingdom and Canada. Physicist J. Robert Oppenheimer directed the Los Alamos Laboratory.", "source_lang": "en"}
    ],
    "hi": [
        {"doc_id": "hi_ld_01", "title": "RAG वास्तुकला परिचय", "text": "रिट्रीवल-ऑगमेंटेड जनरेशन तेज़ वेक्टर खोज और प्रासंगिक जानकारी को एक साथ जोड़ती है। यह एआई मॉडल को सटीक उत्तर देने में सक्षम बनाती है।", "source_lang": "hi"}
    ],
    "mr": [
        {"doc_id": "mr_ld_01", "title": "RAG सिस्टीम माहिती", "text": "RAG प्रणाली वेक्टर सर्चच्या मदतीने माहिती शोधते आणि अचूक उत्तरे तयार करते.", "source_lang": "mr"}
    ]
}


def generate_data():
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    passages_by_lang = {
        "en": ENGLISH_PASSAGES,
        "hi": HINDI_PASSAGES,
        "mr": MARATHI_PASSAGES,
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
        with open(longdoc_path, "w", encoding="utf-8") as f:
            for doc in LONGDOCS.get(lang, []):
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info("Wrote %d longdocs to %s", len(LONGDOCS.get(lang, [])), longdoc_path)
        
    logger.info("Building FAISS HNSW indexes and centroids...")
    idx_mgr = get_index_manager()
    idx_mgr.build_all_indexes()
    logger.info("All FAISS indexes and centroids generated successfully!")

if __name__ == "__main__":
    generate_data()
