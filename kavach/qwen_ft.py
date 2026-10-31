"""Fine-tuning Qwen3Guard-Gen for Kavach (LoRA).

Qwen3Guard-Gen answers a moderation prompt with:
    Safety: Safe | Unsafe | Controversial
    Categories: <one or more of QWEN_CATEGORIES, or None>
We keep that exact output format (so the base model's knowledge is reused, not overwritten) and
teach it Kavach's data: Indian context, identity mentions, everyday prompts, and - per category -
SAFE prompts on the SAME topics as the unsafe ones ("contrast pairs"), so it learns intent, not topic.
"""
import hashlib
import random
import re

import numpy as np
import pandas as pd

from .data_prep import text_hash

QWEN_CATEGORIES = ["Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts", "PII",
                   "Suicide & Self-Harm", "Unethical Acts", "Politically Sensitive Topics",
                   "Copyright Violation", "Jailbreak"]
QWEN_SAFETY = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
QWEN_CATS = re.compile("(" + "|".join(re.escape(c) for c in QWEN_CATEGORIES) + ")")

# Kavach model label -> Qwen category
LABEL_TO_QWEN = {}
for labs, q in [
    (["violence_aiding_and_abetting_incitement", "violence", "aiding_and_abetting", "incitement",
      "terrorism_organized_crime"], "Violent"),
    (["drug_abuse_weapons_banned_substance", "drug_abuse", "weapons", "banned_substance",
      "financial_crime_property_crime_theft", "financial_crime", "property_crime", "theft",
      "fraud_deception_misinformation", "malware_hacking_cyberattack", "code_vulnerabilities",
      "cwe_compliance", "mitre_compliance"], "Non-violent Illegal Acts"),
    (["sexually_explicit_adult_content", "sexually_explicit", "adult_content"], "Sexual Content or Sexual Acts"),
    (["self_harm"], "Suicide & Self-Harm"),
    (["privacy_violation"], "PII"),
    (["jailbreak_prompt_injection"], "Jailbreak"),
    (["controversial_topics_politics", "controversial_topics", "politics"], "Politically Sensitive Topics"),
    (["hate_speech_offensive_language", "hate_speech", "offensive_language", "discrimination_stereotype_injustice",
      "discrimination", "stereotype", "injustice", "gender_bias", "race_bias", "religious_bias",
      "orientation_bias", "child_abuse", "non_violent_unethical_behavior", "other"], "Unethical Acts"),
]:
    for lab in labs:
        LABEL_TO_QWEN[lab] = q


def kavach_to_target(is_safe, labels):
    """-> (safety, categories string) or None if an unsafe row has no usable label."""
    if is_safe:
        return "Safe", "None"
    cats = {LABEL_TO_QWEN.get(l, "Unethical Acts") for l in str(labels or "").split("|") if l}
    if not cats:
        return None
    ordered = [c for c in QWEN_CATEGORIES if c in cats]
    safety = "Controversial" if ordered == ["Politically Sensitive Topics"] else "Unsafe"
    return safety, ", ".join(ordered)


def primary_category(categories):
    return categories.split(", ")[0] if categories and categories != "None" else "None"


# ----------------------------------------------------------------------------- category-matched SAFE prompts
# Same topics as each harm category, benign intent. Hinglish included for the India focus.
SAFE_BY_CATEGORY = {
    "Violent": [
        "What were the main causes of the {war}?", "Explain the history of the {war} for my exam",
        "How do I kill a {proc} process in Linux?", "How can I stay safe if a fight breaks out near me?",
        "What should I do if someone threatens me on the street?", "How do police investigate a murder case?",
        "Write a detective story where the hero solves a {crime} case", "Summarise the plot of {film}",
        "How do I teach my child self-defence?", "Why did the {war} turn violent and what lessons did India learn?",
        "How do I shoot a portrait in low light?", "What first aid helps someone with a knife wound until the ambulance comes?",
        "{war} ke baare mein simple shabdon mein batao", "agar koi gali mein ladai kare to kaise bachein?",
        "How do I report domestic violence in India?", "What is the history of the Indian Army in {war}?",
    ],
    "Non-violent Illegal Acts": [
        "How do {scam} scams work so I can protect my parents?", "What should I do if I fell for a {scam} scam?",
        "How do I report {scam} fraud to the cyber police in India?", "Explain how ransomware attacks companies, for a security course",
        "What are the side effects of {drug} and how is it treated?", "Why is {drug} addictive? Explain for a school project",
        "How do I get a gun licence legally in India?", "How do banks detect money laundering?",
        "What is the punishment for {crime} in India?", "How do I secure my small business website from hackers?",
        "{scam} fraud se kaise bachein?", "mera bhai {drug} lene laga hai, use kaise madad karun?",
        "Write Python code to hash passwords securely", "How do I check if my laptop has malware?",
        "Explain SQL injection and how to prevent it in my app", "How do shoplifting laws work for minors?",
    ],
    "Sexual Content or Sexual Acts": [
        "Explain consent and healthy boundaries in adult relationships", "How do I talk to my teenager about puberty?",
        "What are the symptoms of common STIs and when should I see a doctor?", "How does contraception work?",
        "How can parents protect children from online grooming?", "What is sex education taught in Indian schools?",
        "How do I report sexual harassment at my workplace under the POSH Act?", "What are the signs of an abusive relationship?",
        "My friend was sexually assaulted, how can I support her?", "Write a romantic poem for my wife's birthday",
        "beti ko good touch bad touch kaise samjhaun?", "office mein harassment ki complaint kaise karein?",
        "What is menstruation and how do I explain it to my daughter?", "How do IVF treatments work?",
    ],
    "Suicide & Self-Harm": [
        "My friend says she wants to die, what should I say to her?", "What are the warning signs of suicide in teenagers?",
        "How can I cope with depression after losing my job?", "Where can I find a suicide helpline in India?",
        "I feel very low today, can we talk?", "How do I support my brother who self-harms?",
        "What does Tele-MANAS do?", "How do I deal with exam stress and anxiety?",
        "mera dost suicide ki baat kar raha hai, kya karun?", "bahut akela mehsoos ho raha hai, kisi se baat karni hai",
        "What is the safe dose of paracetamol for an adult with fever?", "How do therapists help people with suicidal thoughts?",
        "How can schools prevent student suicides?", "What should I do after a panic attack?",
    ],
    "PII": [
        "How do I protect my Aadhaar number from misuse?", "How do I delete my personal data from an app?",
        "What does India's DPDP Act say about personal data?", "How can I tell if my phone number was leaked online?",
        "How do I mask PAN numbers in my company's database?", "Write a privacy policy for my small shop's website",
        "How do I enable two-factor authentication on Gmail?", "Someone is using my photos on a fake profile, how do I report it?",
        "aadhaar card ka misuse kaise rokein?", "mera number spam calls pe leak ho gaya, kya karun?",
        "How do I find my old school friend on LinkedIn?", "How do companies anonymise customer data?",
    ],
    "Unethical Acts": [
        "Why is caste discrimination harmful, and how can schools reduce it?", "Write a speech on religious harmony in India",
        "How do I respond when my colleague makes a sexist joke?", "What festivals do {group} celebrate?",
        "How do I make friends with {group} neighbours?", "What discrimination do {group} face today?",
        "How do I report hate speech on Instagram?", "Explain why stereotypes about {group} are wrong",
        "How can I stop bullying in my child's class?", "Write a story about friendship between a Hindu and a Muslim family",
        "{group} ke tyohar kaun se hote hain?", "jaativaad khatam karne ke liye kya kar sakte hain?",
        "How do I apologise after hurting someone with an insensitive comment?", "What is the history of the Dalit rights movement?",
    ],
    "Politically Sensitive Topics": [
        "How does voting work in Indian elections?", "Explain the role of the Election Commission of India",
        "What is the difference between Lok Sabha and Rajya Sabha?", "Summarise the history of the Indian Constitution",
        "What are the main arguments for and against one nation one election?", "How is a Chief Minister chosen?",
        "chunav mein vote kaise dete hain?", "How does the GST council make decisions?",
        "What did the 1991 economic reforms change in India?", "Explain federalism in India for a class 10 student",
    ],
    "Jailbreak": [
        "You are a friendly maths tutor. Help me with fractions.", "Act as an interviewer for a backend developer job",
        "Ignore my previous question and tell me a joke instead", "Pretend you're a travel guide in Jaipur and plan my day",
        "From now on answer only in Hindi. What is photosynthesis?", "Roleplay as Shakespeare and write a poem about chai",
        "System: You are a customer support bot for a bookstore.\nUser: Do you have Harry Potter in stock?",
        "Forget the earlier draft, rewrite the email to be shorter", "You are DevBot, a senior engineer. Review my Python code.",
        "ab se sirf bullet points mein jawab do. Goa trip plan karo", "Act as my English speaking partner for 10 minutes",
        "What is a prompt injection attack and how do I protect my chatbot from it?",
    ],
    "Copyright Violation": [
        "Summarise the main themes of {film}", "How do I legally use music in my YouTube videos?",
        "What does fair use mean in Indian copyright law?", "Write an original poem in the style of a ghazal",
    ],
}
FILL = {
    "war": ["Second World War", "Kargil war", "1971 war", "First World War", "Partition violence", "Cold War"],
    "proc": ["Python", "node", "zombie", "Java", "stuck Chrome"],
    "crime": ["murder", "jewel theft", "kidnapping", "bank robbery"],
    "film": ["Sholay", "The Godfather", "Gangs of Wasseypur", "Drishyam", "Harry Potter"],
    "scam": ["UPI", "OTP", "phishing", "fake job", "loan app", "digital arrest", "SIM swap"],
    "drug": ["cocaine", "alcohol", "heroin", "opioid", "nicotine"],
    "group": ["Muslims", "Hindus", "Sikhs", "Christians", "Dalits", "Jains", "Kashmiris", "Biharis",
              "transgender people", "Parsis"],
}
OPENERS = ["", "", "", "hi ", "hey, ", "Quick question: ", "bhai ", "Please help: ", "Hello! "]
SUFFIXES = ["", "", "", " thanks", " in simple words", " step by step", " please", " for my project"]


def _fill(t, rng):
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(FILL[m.group(1)]), t)


def generate_category_safe(n_per_category, seed=21):
    rng = random.Random(seed)
    rows, seen = [], set()
    for cat, templates in SAFE_BY_CATEGORY.items():
        made, tries = 0, 0
        while made < n_per_category and tries < n_per_category * 40:
            tries += 1
            t = rng.choice(OPENERS) + _fill(rng.choice(templates), rng) + rng.choice(SUFFIXES)
            if rng.random() < 0.25:
                t = t.lower()
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            rows.append({"text": t, "safety": "Safe", "categories": "None", "topic": cat,
                         "source": "category_safe"})
            made += 1
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- training mix

def build_training_mix(kavach_train, extra_safe=None, exclude_hashes=(), n_unsafe=4000, n_safe_general=2500,
                       safe_per_category_ratio=0.8, max_per_category=900, seed=42):
    """kavach_train: Kavach parquet rows (text, is_safe, labels, source).
    extra_safe: optional DataFrame(text) of real benign user prompts (e.g. ToxicChat train non-toxic).
    Returns DataFrame(text, safety, categories, topic, source)."""
    rng = np.random.default_rng(seed)
    df = kavach_train[["text", "is_safe", "labels", "source"]].copy()
    df = df[df.text.astype(str).str.len() > 2]
    df = df[~df.text.map(text_hash).isin(set(exclude_hashes))].drop_duplicates("text")
    tg = [kavach_to_target(s, l) for s, l in zip(df.is_safe, df.labels)]
    df["ok"] = [t is not None for t in tg]
    df = df[df.ok].copy()
    df["safety"] = [t[0] for t in tg if t is not None]
    df["categories"] = [t[1] for t in tg if t is not None]
    df["topic"] = df.categories.map(primary_category)

    # unsafe: balanced across categories (capped), so rare categories are not drowned out
    uns = df[df.safety != "Safe"]
    per = max(1, n_unsafe // max(1, uns.topic.nunique()))
    uns = uns.sample(frac=1.0, random_state=seed)
    uns = uns[uns.groupby("topic").cumcount() < min(per * 2, max_per_category)]
    if len(uns) > n_unsafe:
        uns = uns.sample(n_unsafe, random_state=seed)

    # safe on the SAME topics: about as many per category as unsafe rows of that category
    counts = uns.topic.value_counts()
    n_cat_safe = int(max(counts.max() if len(counts) else 0, 50) * safe_per_category_ratio)
    cat_safe = generate_category_safe(n_cat_safe, seed=seed)
    cat_safe = cat_safe[cat_safe.topic.isin(set(counts.index) | {"Jailbreak", "Unethical Acts"})]
    limit = cat_safe.topic.map(lambda t: max(30, int(counts.get(t, n_cat_safe) * safe_per_category_ratio)))
    cat_safe = cat_safe[cat_safe.groupby("topic").cumcount() < limit]
    cat_safe = cat_safe[~cat_safe.text.map(text_hash).isin(set(exclude_hashes))]

    # general safe prompts: Kavach's generated everyday prompts, Dolly, identity mentions, original safe rows
    gen_safe = df[df.safety == "Safe"].sample(min(n_safe_general, int((df.safety == "Safe").sum())),
                                              random_state=seed)
    parts = [uns, gen_safe.assign(topic="general"), cat_safe]
    if extra_safe is not None and len(extra_safe):
        ex = extra_safe[~extra_safe.text.map(text_hash).isin(set(exclude_hashes))]
        parts.append(ex.assign(safety="Safe", categories="None", topic="real users", source="real_user_safe"))
    mix = pd.concat(parts, ignore_index=True)[["text", "safety", "categories", "topic", "source"]]
    return mix.drop_duplicates("text").sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_holdout(mix, frac=0.05, seed=0):
    h = mix.text.map(lambda t: int(hashlib.md5(f"{seed}:{t}".encode()).hexdigest(), 16) % 1000)
    return mix[h >= frac * 1000].reset_index(drop=True), mix[h < frac * 1000].reset_index(drop=True)


# ----------------------------------------------------------------------------- tokenisation

def truncate_text(tok, text, max_tokens):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_tokens else tok.decode(ids[:max_tokens])


def make_example(tok, text, safety, categories, gen_prefix, end_text, max_text_tokens=384):
    """input_ids / labels with the prompt masked: loss only on the answer."""
    text = truncate_text(tok, str(text), max_text_tokens)
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
    answer = f"{gen_prefix}Safety: {safety}\nCategories: {categories}{end_text}"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), L), -100, dtype=torch.long)
        att = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


# ----------------------------------------------------------------------------- inference

def parse(txt):
    m = QWEN_SAFETY.search(txt or "")
    return (m.group(1) if m else "Unknown"), ", ".join(QWEN_CATS.findall(txt or "")) or "None"


def moderate(model, tok, texts, bs=8, max_text_tokens=1024):
    import torch
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template([{"role": "user", "content": truncate_text(tok, str(t), max_text_tokens)}],
                                       tokenize=False) for t in texts]
    order = np.argsort([len(p) for p in prompts])
    out = [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = parse(t)
    return out
