"""
Comprehensive Technology & AI Domain Passages for VECTOR Corpus.
Covers Python, Data Science, Math, ML, Deep Learning, GenAI/LLMs, Computer Vision, and MLOps.
"""

EN_TECH_PASSAGES = [
    ("en_p_tech_001", "Python is a high-level programming language widely used in data science, with NumPy providing fast multi-dimensional arrays and Pandas offering DataFrame structures for data manipulation.", 401, 1),
    ("en_p_tech_002", "Data preprocessing cleans raw datasets by handling missing values and outliers, while SQL queries extract data from databases, and Matplotlib creates data visualizations.", 402, 1),
    ("en_p_tech_003", "Git and GitHub enable version control for code collaboration, Linux command line manages server environments, APIs exchange JSON data, and Jupyter Notebooks facilitate interactive data analysis.", 403, 1),
    ("en_p_tech_004", "Linear algebra forms the foundation of machine learning, using vectors and matrices to represent datasets, spatial transformations, and neural network weights.", 404, 1),
    ("en_p_tech_005", "Statistics analyzes mean, median, variance, and standard deviation, while probability theory, correlation, covariance, and Bayes theorem calculate conditional likelihoods in ML models.", 405, 1),
    ("en_p_tech_006", "Calculus basics including derivatives and gradients guide optimization algorithms like gradient descent to minimize loss functions during machine learning model training.", 406, 1),
    ("en_p_tech_007", "Machine learning paradigms include supervised learning with labeled data, unsupervised learning for clustering unlabelled data, and reinforcement learning using reward feedback.", 407, 1),
    ("en_p_tech_008", "Linear regression predicts continuous numeric values, logistic regression performs binary classification, while decision trees and random forests ensemble multiple decision rules.", 408, 1),
    ("en_p_tech_009", "XGBoost provides gradient boosted decision trees, Support Vector Machines (SVM) create maximum-margin classification boundaries, K-Nearest Neighbors (KNN) classifies by proximity, K-Means clusters data, and PCA reduces dimensionality.", 409, 1),
    ("en_p_tech_010", "Feature engineering transforms raw variables, train-validation-test splitting prevents data leakage, cross-validation evaluates generalization, hyperparameter tuning optimizes parameters, and metrics like precision and recall assess ML performance.", 410, 1),
    ("en_p_tech_011", "Neural networks process inputs through multi-layer perceptrons, using activation functions like ReLU and backpropagation with loss gradients to update synaptic weights.", 411, 1),
    ("en_p_tech_012", "Convolutional Neural Networks (CNNs) process spatial images, Recurrent Neural Networks (RNNs) and LSTMs handle sequential text data, and transfer learning reuses pretrained feature extractors.", 412, 1),
    ("en_p_tech_013", "Transformer architectures utilize self-attention mechanisms to weigh relationships between all tokens in a sequence, mapping text into continuous vector embeddings.", 413, 1),
    ("en_p_tech_014", "Large Language Models (LLMs) break text into sub-word tokens, process prompts via prompt engineering, and undergo fine-tuning or parameter-efficient LoRA and QLoRA adaptation.", 414, 1),
    ("en_p_tech_015", "Retrieval-Augmented Generation (RAG) splits text into semantic chunks, stores vectors in vector databases for sub-second semantic search, and reranks candidates before generation.", 415, 1),
    ("en_p_tech_016", "AI agents execute multi-step agentic workflows using tool calling, grounding outputs to prevent hallucinations, with orchestration frameworks like LangChain and LangGraph.", 416, 1),
    ("en_p_tech_017", "Multimodal AI integrates text, vision, speech-to-text (STT), and text-to-speech (TTS), evaluating performance through automated LLM benchmark frameworks.", 417, 1),
    ("en_p_tech_018", "Computer vision utilizes OpenCV for image processing, YOLO for real-time object detection, image segmentation for pixel-level masks, and OCR for text extraction.", 418, 1),
    ("en_p_tech_019", "Vision Transformers (ViTs) adapt self-attention mechanisms to image patches, enabling multimodal vision-language models to perform visual question answering and image description.", 419, 1),
    ("en_p_tech_020", "MLOps standardizes machine learning lifecycle management using FastAPI REST endpoints, Docker containerization, cloud model serving, MLflow monitoring, and INT8 quantization for edge optimization.", 420, 1),
]

HI_TECH_PASSAGES = [
    ("hi_p_tech_001", "पायथन डेटा साइंस और मशीन लर्निंग की मुख्य भाषा है, जिसमें NumPy बहु-आयामी ऐरे और Pandas डेटाफ्रेम के माध्यम से डेटा विश्लेषण और प्रीप्रोसेसिंग को सरल बनाते हैं।", 501, 1),
    ("hi_p_tech_002", "SQL डेटाबेस से डेटा निकालने के लिए उपयोग किया जाता है, Git और GitHub कोड वर्जन कंट्रोल प्रदान करते हैं, और Jupyter नोटबुक में इंटरैक्टिव कोड निष्पादित किया जाता है।", 502, 1),
    ("hi_p_tech_003", "रैखिक बीजगणित (लीनियर अलजेब्रा), सांख्यिकी (माध्य, मानक विचलन) और प्रायिकता (बायस थ्योरम) मशीन लर्निंग मॉडल और डेटा विश्लेषण के गणितीय आधार हैं।", 503, 1),
    ("hi_p_tech_004", "कलन (कैलकुलस) और ग्रेडिएंट डिसेंट मॉडल के नुकसान (लॉस) को कम करते हैं, जबकि सुपरवाइज्ड और अनसुपरवाइज्ड लर्निंग डेटा से पैटर्न सीखती हैं।", 504, 1),
    ("hi_p_tech_005", "लीनियर रिग्रेशन, डिसीजन ट्री, रैंडम फॉरेस्ट और XGBoost मुख्य एल्गोरिदम हैं, और PCA डेटा के आयाम (डायमेंशन) को कम करने के लिए उपयोग किया जाता है।", 505, 1),
    ("hi_p_tech_006", "क्रॉस-वैलिडेशन और हाइपरपैरामीटर ट्यूनिंग मॉडल को ओवरफिटिंग से बचाते हैं, जबकि प्रिसिजन, रिकॉल और F1-स्कोर से मॉडल के प्रदर्शन का मूल्यांकन किया जाता है।", 506, 1),
    ("hi_p_tech_007", "न्यूरल नेटवर्क बैकप्रोपैगेशन और एक्टिवेशन फंक्शन का उपयोग करते हैं, CNN इमेज प्रोसेसिंग के लिए, और ट्रांसफॉर्मर (Transformer) मॉडल प्राकृतिक भाषा (NLP) के लिए सर्वश्रेष्ठ हैं।", 507, 1),
    ("hi_p_tech_008", "लार्ज लैंग्वेज मॉडल (LLM) प्रॉम्प्ट इंजीनियरिंग, टोकनाइजेशन और LoRA फाइन-ट्यूनिंग से काम करते हैं, और RAG तकनीक वेक्टर डेटाबेस से प्रासंगिक जानकारी ढूंढती है।", 508, 1),
    ("hi_p_tech_009", "एआई एजेंट्स टूल कॉलिंग और LangChain के माध्यम से जटिल कार्य करते हैं, और मल्टीमॉडल एआई स्पीच-टू-टेक्स्ट (STT) और विजन को एक साथ जोड़ता है।", 509, 1),
    ("hi_p_tech_010", "कंप्यूटर विजन और OpenCV इमेज प्रोसेसिंग करते हैं, YOLO रियल-टाइम ऑब्जेक्ट डिटेक्शन करता है, और OCR छवियों से पाठ (टेक्स्ट) निकालता है।", 510, 1),
    ("hi_p_tech_011", "MLOps में FastAPI से REST API बनाई जाती है, Docker से मॉडल को कंटेनर में डाला जाता है, और क्वांटाइजेशन (Quantization) से मॉडल को हल्का और तेज़ बनाया जाता है।", 511, 1),
]

MR_TECH_PASSAGES = [
    ("mr_p_tech_001", "पायथन ही डेटा सायन्स आणि AI मधील मुख्य भाषा आहे. NumPy आणि Pandas च्या मदतीने डेटाचे विश्लेषण आणि प्रीप्रोसेसिंग केले जाते.", 601, 1),
    ("mr_p_tech_002", "SQL द्वारे डेटाबेस माहिती मिळवली जाते, Git आणि GitHub कोड व्हर्जन कंट्रोलसाठी वापरले जातात, आणि Jupyter मध्ये डेटा सायन्स प्रयोग केले जातात.", 602, 1),
    ("mr_p_tech_003", "लीनियर अलजेब्रा, सांख्यिकी (माध्य, प्रमाण विचलन) आणि कॅल्क्युलस ग्रेडिएंट्स मशीन लर्निंग मॉडेल्सचा पाया आहेत.", 603, 1),
    ("mr_p_tech_004", "सुपरवाइज्ड आणि अनसुपरवाइज्ड लर्निंग, लीनियर रिग्रेशन, डिसीजन ट्री, रँडम फॉरेस्ट आणि XGBoost हे मशीन लर्निंगचे मुख्य प्रकार आहेत.", 604, 1),
    ("mr_p_tech_005", "फीचर इंजिनिअरिंग आणि हायपरपॅरामीटर ट्यूनिंगमुळे मॉडेल ओव्हरफिटींगपासून वाचते आणि योग्य निकाल देते.", 605, 1),
    ("mr_p_tech_006", "न्यूरल नेटवर्क्स आणि बॅकप्रोपॅगेशन डीप लर्निंगचा आधार आहेत, तर ट्रान्सफॉर्मर्स आणि अटेंशन मेकॅनिझम भाषा प्रक्रिया (NLP) सुलभ करतात.", 606, 1),
    ("mr_p_tech_007", "लार्ज लँग्वेज मॉडेल्स (LLMs), प्रॉम्प्ट इंजिनिअरिंग, LoRA फाइन-ट्यूनिंग आणि RAG सिस्टीम वेक्टर डेटाबेसच्या मदतीने अचूक उत्तरे देतात.", 607, 1),
    ("mr_p_tech_008", "AI एजंट्स टूल कॉलिंग आणि LangChain द्वारे स्वयंचलित कामे करतात, तर मल्टीमॉडल AI टेक्स्ट, व्हॉईस आणि इमेज एकत्र हाताळते.", 608, 1),
    ("mr_p_tech_009", "संगणक व्हिजन (Computer Vision), OpenCV आणि YOLO ऑब्जेक्ट डिटेक्ट करतात आणि OCR चित्रातील मजकूर वाचते.", 609, 1),
    ("mr_p_tech_010", "MLOps मध्ये FastAPI आणि Docker द्वारे मॉडेल्स क्लाउडवर तैनात (Deploy) केले जातात, आणि क्वांटायझेशनने (Quantization) मॉडेलचा वेग वाढवला जातो.", 610, 1),
]

TECH_LONGDOCS = {
    "en": [
        {"doc_id": "en_tech_ld_01", "title": "Data Science & Python Ecosystem", "text": "Python forms the backbone of modern data science and AI development. With libraries like NumPy for array operations, Pandas for structured data manipulation, and Matplotlib for data visualization, developers preprocess raw data and compute statistical metrics such as mean, variance, and correlation. Version control using Git and GitHub ensures reproducible workflows, while SQL enables database querying.", "source_lang": "en"},
        {"doc_id": "en_tech_ld_02", "title": "Machine Learning Algorithms & Pipeline", "text": "Machine learning comprises supervised, unsupervised, and reinforcement learning. Algorithms range from linear and logistic regression to decision trees, random forests, XGBoost, SVM, KNN, K-Means clustering, and PCA. The ML pipeline includes feature engineering, handling missing values and outliers, train-test splitting, cross-validation, hyperparameter tuning, and metric evaluation (precision, recall, F1-score) to prevent overfitting.", "source_lang": "en"},
        {"doc_id": "en_tech_ld_03", "title": "Deep Learning & Transformer Models", "text": "Deep learning uses neural networks with multi-layer perceptrons, activation functions like ReLU, and backpropagation optimization. Convolutional Neural Networks (CNNs) handle spatial computer vision, while Recurrent Neural Networks (RNNs) and LSTMs process sequential data. Transformer architectures with self-attention mechanisms enable state-of-the-art Natural Language Processing (NLP) and dense vector embeddings.", "source_lang": "en"},
        {"doc_id": "en_tech_ld_04", "title": "Generative AI, LLMs, RAG & Autonomous Agents", "text": "Generative AI focuses on Large Language Models (LLMs) trained on tokenized text. Technique like prompt engineering, fine-tuning, LoRA, and QLoRA adapt models to domain tasks. Retrieval-Augmented Generation (RAG) utilizes semantic chunking, vector databases, and reranking to eliminate hallucinations. Autonomous AI agents leverage tool calling, agentic workflows, LangChain, and LangGraph to perform complex multi-step reasoning.", "source_lang": "en"},
        {"doc_id": "en_tech_ld_05", "title": "Computer Vision & MLOps Infrastructure", "text": "Computer Vision leverages OpenCV for image processing, YOLO for real-time object detection, segmentation, and OCR for text extraction. MLOps standardizes deployment using FastAPI REST APIs, Docker containers, cloud model serving, MLflow monitoring, and INT8 quantization for edge model optimization.", "source_lang": "en"},
    ],
    "hi": [
        {"doc_id": "hi_tech_ld_01", "title": "डेटा साइंस, मशीन लर्निंग और एआई परिचय", "text": "पायथन, NumPy और Pandas डेटा विश्लेषण और प्रीप्रोसेसिंग के आधार हैं। लीनियर रिग्रेशन, डिसीजन ट्री, रैंडम फॉरेस्ट और XGBoost मशीन लर्निंग के प्रमुख मॉडल हैं। न्यूरल नेटवर्क, बैकप्रोपैगेशन और ट्रांसफॉर्मर मॉडल डीप लर्निंग और भाषा प्रसंस्करण को गति देते हैं।", "source_lang": "hi"},
        {"doc_id": "hi_tech_ld_02", "title": "जनरेटिव एआई, RAG और कंप्यूटर विजन", "text": "लार्ज लैंग्वेज मॉडल (LLM), LoRA फाइन-ट्यूनिंग और RAG तकनीकें वेक्टर डेटाबेस के साथ मिलकर काम करती हैं। एआई एजेंट्स स्वयंचलित कार्य करते हैं। कंप्यूटर विजन में OpenCV और YOLO ऑब्जेक्ट डिटेक्ट करते हैं, और MLOps में FastAPI और Docker से मॉडल तैनात (Deploy) होते हैं।", "source_lang": "hi"},
    ],
    "mr": [
        {"doc_id": "mr_tech_ld_01", "title": "डेटा सायन्स, मशीन लर्निंग आणि AI प्रणाली", "text": "पायथन, NumPy आणि Pandas च्या मदतीने डेटा प्रीप्रोसेसिंग केले जाते. मशीन लर्निंगमध्ये रिग्रेशन, डिसीजन ट्री, रँडम फॉरेस्ट आणि XGBoost वापरले जातात. न्यूरल नेटवर्क्स आणि ट्रान्सफॉर्मर्स डीप लर्निंगचा पाया आहेत.", "source_lang": "mr"},
        {"doc_id": "mr_tech_ld_02", "title": "जनरेटिव्ह AI, RAG आणि MLOps तंत्रज्ञान", "text": "LLMs, प्रॉम्प्ट इंजिनिअरिंग, RAG आणि वेक्टर डेटाबेस माहिती शोधणे सुलभ करतात. AI एजंट्स टूल कॉलिंगद्वारे स्वयंचलित कामे करतात. OpenCV आणि YOLO कॉम्प्युटर व्हिजनसाठी, तर FastAPI आणि Docker MLOps मॉडेल डिप्लॉयमेंटसाठी वापरले जातात.", "source_lang": "mr"},
    ],
}
