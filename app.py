import streamlit as st
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import torch

@st.cache_resource

# generate model annd tokenns 
def load_model():
    
    model_name = "facebook/nllb-200-distilled-600M"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return tokenizer, model


tokenizer, model = load_model()


# Translation function

def translate(text, tokenizer, model, source_lang_code, target_lang_code):
    # Tell the tokenizer what is the language user is typing 
    tokenizer.src_lang = source_lang_code
    
    # Convert the text inot mathematical tensor token 
    inputs = tokenizer(text, return_tensors="pt", padding = True, truncation = True)
    
    # get the exact ID for the target language 
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id = target_lang_id,
            max_length = 400
        )
        
    return tokenizer.batch_decode(translated_tokens,skip_special_tokens = True)[0]


st.set_page_config(page_title = "Language Translator", page_icon="🌍")
st.title("LANGUAGE TRANSLATOR")

# languages.py
languages = {
    "Acehnese (Arabic)": "ace_Arab", "Acehnese (Latin)": "ace_Latn", "Afrikaans": "afr_Latn", "Akan": "aka_Latn",
    "Amharic": "amh_Ethi", "Armenian": "hye_Armn", "Assamese": "asm_Beng", "Asturian": "ast_Latn",
    "Ayacucho Quechua": "quy_Latn", "Azerbaijani": "azj_Latn", "Balinese": "ban_Latn", "Bambara": "bam_Latn",
    "Basque": "eus_Latn", "Belarusian": "bel_Cyrl", "Bemba": "bem_Latn", "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva", "Bosnian": "bos_Latn", "Buginese": "bug_Bugt", "Bulgarian": "bul_Cyrl",
    "Burmese": "mya_Mymr", "Catalan": "cat_Latn", "Cebuano": "ceb_Latn", "Central Atlas Tamazight": "tzm_Tfng",
    "Central Aymara": "ayr_Latn", "Central Kanuri (Arabic)": "knc_Arab", "Central Kanuri (Latin)": "knc_Latn",
    "Central Kurdish": "ckb_Arab", "Chhattisgarhi": "hne_Deva", "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant", "Crimean Tatar": "crh_Latn", "Croatian": "hrv_Latn", "Czech": "ces_Latn",
    "Danish": "dan_Latn", "Dari": "prs_Arab", "Dutch": "nld_Latn", "Dyula": "dyu_Latn", "Dzongkha": "dzo_Tibt",
    "English": "eng_Latn", "Esperanto": "epo_Latn", "Estonian": "est_Latn", "Ewe": "ewe_Latn", 
    "Faroese": "fao_Latn", "Fijian": "fij_Latn", "Filipino": "tgl_Latn", "Finnish": "fin_Latn",
    "Fon": "fon_Latn", "French": "fra_Latn", "Friulian": "fur_Latn", "Galician": "glg_Latn", 
    "Ganda": "lug_Latn", "Georgian": "kat_Geor", "German": "deu_Latn", "Greek": "ell_Grek", 
    "Guarani": "grn_Latn", "Gujarati": "guj_Gujr", "Haitian Creole": "hat_Latn", "Hausa": "hau_Latn", 
    "Hebrew": "heb_Hebr", "Hindi": "hin_Deva", "Hungarian": "hun_Latn", "Icelandic": "isl_Latn", 
    "Igbo": "ibo_Latn", "Ilocano": "ilo_Latn", "Indonesian": "ind_Latn", "Irish": "gle_Latn", 
    "Italian": "ita_Latn", "Japanese": "jpn_Jpan", "Javanese": "jav_Latn", "Kabuverdianu": "kea_Latn", 
    "Kabyle": "kab_Latn", "Kamba": "kam_Latn", "Kannada": "kan_Knda", "Kashmiri (Arabic)": "kas_Arab", 
    "Kashmiri (Devanagari)": "kas_Deva", "Kazakh": "kaz_Cyrl", "Khmer": "khm_Khmr", "Kikuyu": "kik_Latn", 
    "Kinyarwanda": "kin_Latn", "Korean": "kor_Hang", "Kyrgyz": "kir_Cyrl", "Lao": "lao_Laoo", 
    "Latvian": "lvs_Latn", "Ligurian": "lij_Latn", "Limburgish": "lim_Latn", "Lingala": "lin_Latn", 
    "Lithuanian": "lit_Latn", "Lombard": "lmo_Latn", "Luba-Kasai": "lua_Latn", "Luo": "luo_Latn", 
    "Luxembourgish": "ltz_Latn", "Macedonian": "mkd_Cyrl", "Magahi": "mag_Deva", "Maithili": "mai_Deva", 
    "Malagasy": "mlg_Latn", "Malay": "zsm_Latn", "Malayalam": "mal_Mlym", "Maltese": "mlt_Latn", 
    "Maori": "mri_Latn", "Marathi": "mar_Deva", "Minangkabau (Arabic)": "min_Arab", "Minangkabau (Latin)": "min_Latn", 
    "Modern Standard Arabic": "arb_Arab", "Mongolian": "khk_Cyrl", "Mossi": "mos_Latn", "Nepali": "npi_Deva", 
    "Nigerian Fulfulde": "fuv_Latn", "North Azerbaijani": "azj_Latn", "Northern Kurdish": "kmr_Latn", 
    "Northern Sotho": "nso_Latn", "Norwegian Bokmål": "nob_Latn", "Norwegian Nynorsk": "nno_Latn", 
    "Nuer": "nus_Latn", "Nyanja": "nya_Latn", "Occitan": "oci_Latn", "Odia": "ory_Orya", 
    "Pangasinan": "pag_Latn", "Papiamento": "pap_Latn", "Pashto": "pbt_Arab", "Persian": "pes_Arab", 
    "Polish": "pol_Latn", "Portuguese": "por_Latn", "Punjabi (Gurmukhi)": "pan_Guru", "Romanian": "ron_Latn", 
    "Rundi": "run_Latn", "Russian": "rus_Cyrl", "Samoan": "smo_Latn", "Sango": "sag_Latn", 
    "Sanskrit": "san_Deva", "Sardinian": "srd_Latn", "Scottish Gaelic": "gla_Latn", "Serbian": "srp_Cyrl", 
    "Shona": "sna_Latn", "Sicilian": "scn_Latn", "Silesian": "szl_Latn", "Sindhi": "snd_Arab", 
    "Sinhala": "sin_Sinh", "Slovak": "slk_Latn", "Slovenian": "slv_Latn", "Somali": "som_Latn", 
    "Southern Sotho": "sot_Latn", "Spanish": "spa_Latn", "Sundanese": "sun_Latn", "Swahili": "swh_Latn", 
    "Swati": "ssw_Latn", "Swedish": "swe_Latn", "Tagalog": "tgl_Latn", "Tajik": "tgk_Cyrl", 
    "Tamil": "tam_Taml", "Tatar": "tat_Cyrl", "Telugu": "tel_Telu", "Thai": "tha_Thai", 
    "Tibetan": "bod_Tibt", "Tigrinya": "tir_Ethi", "Tok Pisin": "tpi_Latn", "Tsonga": "tso_Latn", 
    "Tswana": "tsn_Latn", "Turkish": "tur_Latn", "Turkmen": "tuk_Latn", "Twi": "twi_Latn", 
    "Ukrainian": "ukr_Cyrl", "Urdu": "urd_Arab", "Uzbek": "uzn_Latn", "Venetian": "vec_Latn", 
    "Vietnamese": "vie_Latn", "Welsh": "cym_Latn", "Wolof": "wol_Latn", "Xhosa": "xho_Latn", 
    "Yiddish": "ydd_Hebr", "Yoruba": "yor_Latn", "Zulu": "zul_Latn"
}

lang_names = list(languages.keys())

col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("Translate From:",lang_names, index=39)
with col2:
    target_lang_name = st.selectbox("Translate To:",lang_names, index=39)
    

source_code = languages[source_lang_name]
target_code = languages[target_lang_name]

# 5. The input text box
source_text = st.text_area("Enter text to translate:", height=150)

# 6. The Translation Button & Execution Logic
if st.button("Translate", type="primary"):
    if source_text.strip():
        with st.spinner(f"Translating to {target_lang_name}..."):
            # Call the translate function you wrote in Step 2
            translation = translate(source_text, tokenizer, model, source_code, target_code)
            
            # Display the result
            st.success("Complete!")
            st.info(translation)
    else:
        st.warning("Please enter some text first.")