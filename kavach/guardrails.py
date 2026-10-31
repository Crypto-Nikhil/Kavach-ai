"""Customer-facing guardrails.

The model predicts ~40 fine-grained labels inherited from its training sources (several are
duplicates or fragments of each other, e.g. `violence`, `incitement` and
`violence_aiding_and_abetting_incitement`). Customers should not see that. They control a
short list of clear guardrails; each guardrail owns a set of model labels.

Any model label not listed here falls into the `other_harmful` guardrail automatically, so a
retrained model with new labels never silently escapes the policy.
"""

GUARDRAILS = [
    {"id": "jailbreak", "name": "Jailbreaks & prompt injection",
     "description": "Attempts to override the AI's instructions, extract its system prompt or unlock 'no rules' modes.",
     "labels": ["jailbreak_prompt_injection"]},
    {"id": "violence", "name": "Violence & incitement",
     "description": "Planning, encouraging or helping with physical harm to people.",
     "labels": ["violence_aiding_and_abetting_incitement", "violence", "aiding_and_abetting", "incitement"]},
    {"id": "hate", "name": "Hate speech & harassment",
     "description": "Slurs, insults and abusive language aimed at people or groups.",
     "labels": ["hate_speech_offensive_language", "hate_speech", "offensive_language"]},
    {"id": "discrimination", "name": "Discrimination & bias",
     "description": "Stereotyping or demeaning people by gender, race, religion, orientation and similar traits.",
     "labels": ["discrimination_stereotype_injustice", "discrimination", "stereotype", "injustice",
                "gender_bias", "race_bias", "religious_bias", "orientation_bias"]},
    {"id": "sexual", "name": "Sexual content",
     "description": "Sexually explicit or adult content.",
     "labels": ["sexually_explicit_adult_content", "sexually_explicit", "adult_content"]},
    {"id": "child_safety", "name": "Child safety",
     "description": "Any content that sexualises, abuses or endangers children.",
     "labels": ["child_abuse"], "critical": True},
    {"id": "self_harm", "name": "Self-harm & suicide",
     "description": "Content about hurting oneself. Returns a support message with helpline details.",
     "labels": ["self_harm"], "critical": True},
    {"id": "weapons_drugs", "name": "Weapons & drugs",
     "description": "Making or acquiring weapons, drugs or banned substances.",
     "labels": ["drug_abuse_weapons_banned_substance", "drug_abuse", "weapons", "banned_substance"]},
    {"id": "cyber", "name": "Malware & hacking",
     "description": "Malware, cyberattacks and requests for insecure or exploitable code.",
     "labels": ["malware_hacking_cyberattack", "code_vulnerabilities", "cwe_compliance", "mitre_compliance"]},
    {"id": "fraud", "name": "Fraud, scams & misinformation",
     "description": "Scams, impersonation, deception and deliberate misinformation.",
     "labels": ["fraud_deception_misinformation"]},
    {"id": "crime", "name": "Theft & financial crime",
     "description": "Stealing, money laundering and other property or financial crimes.",
     "labels": ["financial_crime_property_crime_theft", "financial_crime", "property_crime", "theft"]},
    {"id": "privacy", "name": "Privacy violations",
     "description": "Doxxing, stalking, or getting into someone else's accounts or personal data.",
     "labels": ["privacy_violation"]},
    {"id": "terrorism", "name": "Terrorism & organised crime",
     "description": "Support for or planning of terrorism and organised crime.",
     "labels": ["terrorism_organized_crime"]},
    {"id": "politics", "name": "Politics & controversial topics",
     "description": "Politically charged or controversial discussion. Often allowed by news and education apps.",
     "labels": ["controversial_topics_politics", "controversial_topics", "politics"]},
    {"id": "other_harmful", "name": "Other unethical behaviour",
     "description": "Harmful requests that fit no other guardrail.",
     "labels": ["non_violent_unethical_behavior", "other"]},
]

GUARDRAIL_BY_ID = {g["id"]: g for g in GUARDRAILS}

SUPPORT_MESSAGE = (
    "It sounds like you might be going through something really difficult. You don't have to "
    "face it alone. In India you can call Tele-MANAS on 14416 (free, 24x7). Outside India, "
    "findahelpline.com lists free helplines in your country. If you are in immediate danger, "
    "please contact local emergency services."
)

PRESETS = {
    "Strict (default)": {"default_action": "block", "guardrails": {}},
    "Balanced": {"default_action": "block", "guardrails": {"politics": "flag", "other_harmful": "flag",
                                                           "sexual": "flag"}},
    "News & media": {"default_action": "block", "guardrails": {"politics": "allow", "discrimination": "flag",
                                                               "other_harmful": "flag"}},
    "Kids & education": {"default_action": "block", "guardrails": {}, "sensitivity": "strict"},
    "Security research": {"default_action": "block", "guardrails": {"cyber": "flag", "jailbreak": "flag"}},
}


def label_to_guardrail(model_labels):
    """Map every model label to a guardrail id (unknown labels -> other_harmful)."""
    owner = {}
    for g in GUARDRAILS:
        for lab in g["labels"]:
            owner[lab] = g["id"]
    return {lab: owner.get(lab, "other_harmful") for lab in model_labels}
