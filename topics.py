"""
Curated list of health topics to pull from MedlinePlus.
Feel free to add/remove terms — each one becomes a query against the
MedlinePlus Web Service (https://wsearch.nlm.nih.gov/ws/query).
"""

HEALTH_TOPICS = [
    # Chronic / metabolic
    "diabetes", "type 2 diabetes", "hypertension", "high cholesterol",
    "obesity", "metabolic syndrome", "hypothyroidism", "hyperthyroidism",
    "gout", "osteoporosis",

    # Cardiovascular
    "heart disease", "heart failure", "coronary artery disease",
    "stroke", "arrhythmia", "atrial fibrillation", "peripheral artery disease",
    "deep vein thrombosis", "varicose veins",

    # Respiratory
    "asthma", "copd", "pneumonia", "bronchitis", "tuberculosis",
    "sleep apnea", "influenza", "common cold", "covid-19", "sinusitis",

    # Infectious disease
    "hiv/aids", "hepatitis", "hepatitis b", "hepatitis c", "malaria",
    "lyme disease", "measles", "chickenpox", "shingles", "urinary tract infection",
    "strep throat", "mononucleosis", "meningitis",

    # Mental health
    "depression", "anxiety disorders", "bipolar disorder", "schizophrenia",
    "ptsd", "ocd", "adhd", "eating disorders", "insomnia", "panic disorder",

    # Neurological
    "migraine", "epilepsy", "parkinson's disease", "alzheimer's disease",
    "multiple sclerosis", "bell's palsy", "vertigo", "peripheral neuropathy",

    # Digestive
    "gerd", "acid reflux", "irritable bowel syndrome", "crohn's disease",
    "ulcerative colitis", "celiac disease", "gallstones", "pancreatitis",
    "constipation", "diverticulitis", "peptic ulcer",

    # Musculoskeletal
    "arthritis", "rheumatoid arthritis", "osteoarthritis", "fibromyalgia",
    "back pain", "carpal tunnel syndrome", "tendinitis", "sciatica",

    # Skin
    "acne", "eczema", "psoriasis", "rosacea", "dermatitis", "hives",
    "fungal skin infection", "cellulitis",

    # Cancer
    "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer",
    "skin cancer", "leukemia", "lymphoma", "cervical cancer",

    # Women's / reproductive health
    "pregnancy", "menopause", "endometriosis", "pcos",
    "urinary incontinence", "fibroids",

    # Kidney / urinary
    "kidney stones", "chronic kidney disease", "kidney infection",

    # Eye / ear
    "cataracts", "glaucoma", "macular degeneration", "conjunctivitis",
    "ear infection", "tinnitus", "hearing loss",

    # Allergy / immune
    "allergies", "food allergy", "anaphylaxis", "lupus", "eczema",

    # Endocrine / blood
    "anemia", "vitamin d deficiency", "vitamin b12 deficiency",

    # Pediatric
    "croup", "hand foot and mouth disease", "colic",

    # General / injury
    "concussion", "fractures", "dehydration", "heat exhaustion",
    "food poisoning", "high blood pressure", "chronic pain",
]
