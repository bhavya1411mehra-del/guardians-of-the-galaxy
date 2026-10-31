"""Evidence-based *public health information* (NOT individual medical advice).

The WHO 2021 PM2.5 guideline of 15 ug/m3 is a **24-hour mean**,
NOT an hourly cut-off and NOT a guarantee of safety below it.
"""

WHO_QA = 'https://www.who.int/news-room/questions-and-answers/item/air-pollution-personal-interventions-and-risk-communication'
WHO_BRIEFS = 'https://www.who.int/europe/publications/m/item/personal-level-actions-to-reduce-air-pollution-exposure'
WHO_CHILD = 'https://www.who.int/publications/i/WHO-CED-PHE-18-01'
WHO_GUIDELINE = 'https://www.ncbi.nlm.nih.gov/books/NBK574582/table/fm-ch1.tab1/'
BMJ_STUDY = 'https://www.bmj.com/content/384/bmj-2023-076322'

SOURCES = [
    ('WHO — Personal interventions / risk communication (2024)', WHO_QA,
     'Discusses adapting exercise timing/location; cautions against treating respirators as a universal solution.'),
    ('WHO — Practical exposure reduction briefs (2024)', WHO_BRIEFS,
     'Reviews reducing exposure time, adjusting activity timing, air cleaners and respirators.'),
    ('WHO — Air pollution and child health (2018)', WHO_CHILD,
     'Documents risks for children and developing lungs.'),
    ('WHO — Global air quality guidelines (2021)', WHO_GUIDELINE,
     'PM2.5 guidance: 24-hour mean 15 ug/m3; annual mean 5 ug/m3.'),
    ('BMJ — Short-term PM2.5 and hospital admissions (2024)', BMJ_STUDY,
     'Large observational time-series study linking short-term exposure to respiratory and cardiovascular outcomes.'),
]

GROUPS = {
    'Everyone': (
        'Avoid unnecessary exposure when air pollution is elevated',
        ['Follow official local air-quality alerts and measured pollutant information.',
         'When practical, shift strenuous outdoor activities to a time/place with lower pollution.',
         'Reducing sources of pollution is more effective at a population level than relying only on individual precautions.'],
        [WHO_QA, WHO_BRIEFS]),
    'Children & teenagers': (
        'Children need particular consideration',
        ['Children have developing lungs and may be more susceptible to air-pollution harms.',
         'On pollution episodes, consider moving vigorous outdoor play to lower-exposure periods or locations.',
         'For children with symptoms or a diagnosed condition, follow their clinician’s advice.'],
        [WHO_CHILD, WHO_QA]),
    'Older adults': (
        'Consider cumulative exposure and underlying conditions',
        ['Consider adapting the timing and intensity of outdoor activity during high pollution.',
         'People with heart or lung conditions should follow their existing medical guidance.',
         'Do not stop prescribed medications based on a website forecast.'],
        [WHO_QA, WHO_BRIEFS]),
    'Asthma / COPD': (
        'Air pollution may aggravate respiratory conditions',
        ['Follow the asthma/COPD management plan provided by your clinician.',
         'Consider limiting vigorous outdoor activity during pollution episodes.',
         'Seek medical attention for worsening or severe symptoms; do not rely on app predictions for treatment.'],
        [WHO_QA, BMJ_STUDY]),
    'Heart conditions': (
        'Higher PM2.5 exposure is associated with cardiovascular harm',
        ['Discuss activity limitations with your healthcare professional if you have heart disease.',
         'Where practical, avoid heavy outdoor exertion during pollution episodes.',
         'New or significant chest pain or severe breathlessness warrants urgent medical care.'],
        [BMJ_STUDY, WHO_QA]),
    'Pregnancy': (
        'Take care to minimize avoidable exposure',
        ['Use local air-quality reports when planning unavoidable outdoor travel.',
         'Consider moving strenuous outdoor activities to less-polluted places or times.',
         'Discuss individual pregnancy-related concerns with your prenatal care provider.'],
        [WHO_BRIEFS, WHO_QA]),
    'Outdoor workers': (
        'Long outdoor periods may increase exposure',
        ['When feasible, arrange work and strenuous tasks for relatively lower-pollution hours.',
         'Use employer and occupational-health guidance; workplace respirator requirements differ from advice for the general public.',
         'Do not treat any forecast hour as exposure-free.'],
        [WHO_QA, WHO_BRIEFS]),
}


def guidance(group: str):
    return GROUPS[group]


def reference_for(url: str):
    for title, link, summary in SOURCES:
        if link == url:
            return title, summary
    return url, ''


def pm25_24h_mean(history, predicted=None):
    """Compute a measured or modeled history mean only from 24 consecutive hourly samples."""
    import pandas as pd
    series = history[['time', 'pm2_5']].copy().sort_values('time').drop_duplicates('time')
    latest = series.tail(24)
    if len(latest) != 24:
        return None
    if (pd.to_datetime(latest['time']).diff().dropna() != pd.Timedelta(hours=1)).any():
        return None
    if latest['pm2_5'].isna().any():
        return None
    return float(latest['pm2_5'].mean())
