import os
import sys
import cohere
import json
import random
import joblib
import logging
import numpy as np
import pandas as pd


from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func

from sqlalchemy import func
from flask import (
    Blueprint, render_template, request, session, jsonify, url_for, redirect
)
from flask_login import login_required, current_user
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

main = Blueprint('main', __name__)

from flask import (
    Flask, render_template, redirect, url_for,
    flash, request, jsonify, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
import requests

from sentence_transformers import SentenceTransformer, util
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── App & DB Setup ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY='dev-key',
    SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(app.root_path, 'app.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
db      = SQLAlchemy(app)
migrate = Migrate(app, db)

# ─── Login Manager ──────────────────────────────────────────────────────────
login = LoginManager(app)
login.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id    = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pwd   = db.Column(db.String(128), nullable=False)
    def set_password(self, pw):    self.pwd = generate_password_hash(pw)
    def check_password(self, pw):  return check_password_hash(self.pwd, pw)

class Message(db.Model):
    __tablename__ = 'message'
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender    = db.Column(db.String(10), nullable=False)
    text      = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), index=True)

@login.user_loader
def load_user(uid):
    return User.query.get(int(uid))

class Result(db.Model):
    __tablename__ = 'results'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    condition  = db.Column(db.String(64), nullable=False)
    diagnosis  = db.Column(db.String(64), nullable=False)
    score      = db.Column(db.Integer, nullable=False)          # new!
    timestamp  = db.Column(db.DateTime, server_default=db.func.now(), index=True)
# app.py (after Result)
class QuestionnaireResponse(db.Model):
    __tablename__     = 'questionnaire_responses'
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    answers_json      = db.Column(db.Text, nullable=False)    # raw answers
    total_score       = db.Column(db.Integer, nullable=False) # sum of all answers
    condition         = db.Column(db.String(64), nullable=False)
    diagnosis         = db.Column(db.String(64), nullable=False)
    engine            = db.Column(db.String(32), nullable=False)
    timestamp         = db.Column(db.DateTime,
                                  server_default=db.func.now(),
                                  index=True)




@login.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ─── Utility: map diagnosis → numeric severity ───────────────────────────────
DIAG_SEVERITY = {
    'None':     0,
    'Mild':     1,
    'Moderate': 2,
    'Severe':   3
}

base = app.root_path

# ─── Intermediate Router ─────────────────────────────────────────
router = joblib.load(os.path.join(base, 'router_retrained.pkl'))
app.logger.info("[Startup] Loaded retrained ML router")

# ─── Rule-based Engine Setup ────────────────────────────────────
with open(os.path.join(base, 'Models', 'mentalhealth.json'), 'r', encoding='utf-8') as f:
    intents = json.load(f)['intents']

_patterns, _tags = [], []
for intent in intents:
    for p in intent.get('patterns', []):
        _patterns.append(p)
        _tags.append(intent['tag'])
_responses = { intent['tag']: intent.get('responses', []) for intent in intents }




embedder = SentenceTransformer('all-MiniLM-L6-v2')
pattern_embeddings = embedder.encode(_patterns, convert_to_tensor=True)

def rule_response(text):
    q_emb = embedder.encode(text, convert_to_tensor=True)
    sims  = util.cos_sim(q_emb, pattern_embeddings)[0]
    best_idx, best_score = int(sims.argmax()), float(sims.max())
    if best_score < 0.3:
        return "I’m not sure I understand—could you tell me more about how you feel?"
    tag = _tags[best_idx]
    return random.choice(_responses[tag])



few_shot = """
You are an empathetic counselor. For each User message:
  1) Acknowledge their exact feeling.
  2) Offer one concrete tip.
  3) Keep it under 30 words.

User: I’m feeling overwhelmed by work deadlines.  
Bot: I’m sorry you’re feeling overwhelmed. Try the Pomodoro Technique: 25 minutes focused work, 5 minutes break.

User: I can’t stop replaying old conversations.  
Bot: I’m sorry you’re stuck in your thoughts. Write down what happened, then close the notebook to let it go.

User: I’m anxious about meeting new people.  
Bot: I’m sorry you’re anxious. Before you go, practice 3 slow breaths and remind yourself you’re capable of kindness.

User: I feel lonely even when I’m surrounded by people.  
Bot: I’m sorry you’re feeling lonely. Could you text one friend now and share how you feel?

User: My mind races at night and I can’t sleep.  
Bot: I’m sorry you can’t sleep. Try a 5-minute body scan: slowly notice each part of your body and relax it.

User: I get so frustrated when small tasks pile up.  
Bot: I’m sorry you’re frustrated. Break tasks into tiny steps—complete one step, then pause and breathe before the next.

User: I’m worried I’ll never get better.  
Bot: I’m sorry you’re worried. Remember one past challenge you overcame and how strong you were.

User: My heart pounds whenever I think of today’s meeting.  
Bot: I’m sorry you’re experiencing that. Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.

User: I feel like nobody understands me.  
Bot: I’m sorry you feel misunderstood. Write a letter expressing your emotions—no one else needs to read it.

User: I’m scared of failing again.  
Bot: I’m sorry you’re scared. Reframe one “failure” as feedback by asking: “What can I learn from this?”

User: I feel guilt over things I can’t change.  
Bot: I’m sorry you feel guilty. Try listing what’s within your control right now and focus on one small step.

User: I’m having trouble concentrating on anything.  
Bot: I’m sorry you’re distracted. Use the 5-4-3-2-1 grounding: name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.

User: I’m overwhelmed by all the news lately.  
Bot: I’m sorry you’re overwhelmed. Limit news checks to twice a day and do a comforting hobby afterward.

User: I feel exhausted but can’t rest.  
Bot: I’m sorry you’re exhausted. Try progressive muscle relaxation: tense each muscle group for 5s, then release.

User: I keep comparing myself to others.  
Bot: I’m sorry you’re comparing. Pause and write three things you appreciate about yourself today.

User: {user_text}  
Bot:
"""



#placeholder cleanup helper 
def clean_placeholders(text: str) -> str:
    return (
        text
        .replace("_comma_", ",")
        .replace("_period_", ".")
        .replace("_questionmark_", "?")
        .replace("_exclamation_", "!")
     
    )








# Questionnaire Definitions
QUESTIONS = [
    {'id':'q1','text':'1. Little interest or pleasure in doing things?'},
    {'id':'q2','text':'2. Feeling down, depressed, or hopeless?'},
    {'id':'q3','text':'3. Trouble sleeping too much or too little?'},
    {'id':'q4','text':'4. Feeling tired or having little energy?'},
    {'id':'q5','text':'5. Poor appetite or overeating?'},
    {'id':'q6','text':'6. Feeling bad about yourself—or that you are a failure?'},
    {'id':'q7','text':'7. Trouble concentrating on things?'},
    {'id':'q8','text':'8. Moving or speaking so slowly—or being fidgety?'},
    {'id':'q9','text':'9. Thoughts that you would be better off dead or of hurting yourself?'},
    {'id':'q10','text':'10. Feeling nervous, anxious, or on edge?'},
    {'id':'q11','text':'11. Not being able to stop or control worrying?'},
    {'id':'q12','text':'12. Worrying too much about different things?'},
    {'id':'q13','text':'13. Trouble relaxing?'},
    {'id':'q14','text':'14. Becoming so restless it’s hard to sit still?'},
    {'id':'q15','text':'15. Becoming easily annoyed or irritable?'},
    {'id':'q16','text':'16. Feeling afraid as if something awful might happen?'},
    {'id':'q17','text':'17. Trouble getting unwanted or intrusive thoughts out of your mind?'},
    {'id':'q18','text':'18. Checking things over and over again?'},
    {'id':'q19','text':'19. Feeling compelled to count things?'},
    {'id':'q20','text':'20. Washing or cleaning things repetitively?'},
    {'id':'q21','text':'21. Becoming upset if objects aren’t arranged “just right”?'},
    {'id':'q22','text':'22. Repeating certain words or phrases mentally?'},
    {'id':'q23','text':'23. Nightmares or unwanted memories of a traumatic event?'},
    {'id':'q24','text':'24. Avoiding reminders of a traumatic event?'},
    {'id':'q25','text':'25. Being “on guard,” watchful, or easily startled?'},
    {'id':'q26','text':'26. Feeling numb or detached from people, activities, or surroundings?'},
    {'id':'q27','text':'27. Guilt or blame after a traumatic event?'}
]
OPTIONS = [
    ('Not at all',              0),
    ('Several days',            1),
    ('More than half the days', 2),
    ('Nearly every day',        3)
]
DOMAIN_RANGES = {
    'Depression': list(range(1,10)),
    'Anxiety':    list(range(10,17)),
    'OCD':        list(range(17,23)),
    'PTSD':       list(range(23,28))
}
def grade_severity(score, max_pts):
    pct = score / max_pts
    if pct < 0.33: return 'Mild'
    if pct < 0.66: return 'Moderate'
    return 'Severe'

# Routes 
@app.route('/', methods=['GET'], endpoint='landing')
def landing():
    return render_template('landingpage.html')

# Authentication Routes 
@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('questionnaire'))
    if request.method == 'POST':
        e,p = request.form['email'], request.form['password']
        if User.query.filter_by(email=e).first():
            flash('Email already registered', 'warning')
            return redirect(url_for('register'))
        u = User(email=e); u.set_password(p)
        db.session.add(u); db.session.commit()
        login_user(u)
        return redirect(url_for('questionnaire'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'], endpoint='login')
def login_view():

    if current_user.is_authenticated:
        return redirect(url_for('landing'))

    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        user     = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('landing'))     
        flash('Invalid credentials', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout_view():
    logout_user()
    return redirect(url_for('landing'))





@app.route('/questionnaire', methods=['GET','POST'])
@login_required
def questionnaire():
    if request.method == 'POST':
        ans = { q['id']: int(request.form[q['id']]) for q in QUESTIONS }

        # Compute raw domain scores & severity
        raw = { d: sum(ans[f"q{i}"] for i in idxs)
                for d, idxs in DOMAIN_RANGES.items() }
        norm = { d: raw[d]/(len(DOMAIN_RANGES[d])*3) for d in raw }
        pred = max(norm, key=norm.get)
        sev  = grade_severity(raw[pred], len(DOMAIN_RANGES[pred])*3)

        total = sum(ans.values())
        session['diagnosis'] = (
            'Mild symptoms'   if total <= (27*0.33)
          else 'Moderate symptoms' if total <= (27*0.66)
          else 'Severe symptoms'
        )

        # Scores for ML routing
        dep, anx, ocd, ptsd = raw['Depression'], raw['Anxiety'], raw['OCD'], raw['PTSD']

        # ML-based engine
        df_in = pd.DataFrame([[dep,anx,ocd,ptsd,total]],
                             columns=['dep_score','anx_score','ocd_score','ptsd_score','total_score'])
        engine = router.predict(df_in)[0]
        session['engine'] = engine
        session['suggestion_shown'] = False
        app.logger.info(f"[Router] Selected engine → {engine}")

        # Per-domain 50% conditions
        domain_thresholds = {'Depression':14,'Anxiety':11,'OCD':9,'PTSD':8}
        domain_thresholds = {'Depression':14,'Anxiety':11,'OCD':9,'PTSD':8}
        domain_thresholds = {
            'Depression': int(0.3 * 27),  # ≈8
            'Anxiety':    int(0.3 * 21),  # ≈6
            'OCD':        int(0.3 * 18),  # ≈5
            'PTSD':       int(0.3 * 15)   # ≈4
        }
        dominant = max(raw, key=raw.get)
        if raw[dominant] < domain_thresholds[dominant]:
            session['condition'] = 'Self-Care Recommended'
        else:
            session['condition'] = dominant
            session['total_score'] = total
                # Persist to results tableS
        r = Result(
            user_id   = current_user.id,
            condition = session['condition'],
            diagnosis = session['diagnosis'],
            score     = total   # <-- save it here
        )
       
        db.session.add(r)
        db.session.commit()
    \
        app.logger.info(f"[Router] Selected condition → {session['condition']}")


        Message.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return redirect(url_for('chat'))

    return render_template('questionnaire.html', questions=QUESTIONS, options=OPTIONS)

with open(os.path.join(base, 'Models', 'retrieval.json'), 'r', encoding='utf-8') as f:
    retr_intents   = json.load(f)['intents']
retr_responses = { it['tag']: it['responses'] for it in retr_intents }
_followups     = retr_responses.pop('followups_retrieval', [])

@app.route('/chat', methods=['GET','POST'])
@login_required
def chat():
    engine    = session.get('engine','generative')
    condition = session.get('condition','Self-Care Recommended')
    diagnosis = session.pop('diagnosis', None)

    # 1) Rule-based suggestions (mild)
    if engine == 'rulebased' and not session.get('suggestion_shown', False):
        pred_msg = f"You might have **{condition}**. Here are some self-care tips:"
        greeting = random.choice(_responses.get('suggestion_greeting', []))
        tag = {
            'Depression': 'suggestion_depression',
            'Anxiety':    'suggestion_anxiety',
            'OCD':        'suggestion_OCD',
            'PTSD':       'suggestion_PTSD',
            'Self-Care Recommended': 'suggestion_SelfCareRecommended'
        }[condition]
        tips = _responses.get(tag, [])
        session['suggestion_shown'] = True
        return render_template(
            'chat.html',
            suggestion_flow=True,
            pred_msg=pred_msg,
            greeting=greeting,
            tips=tips,
            diagnosis=diagnosis,
            engine=engine
        )

    # 2) Retrieval-questionnaire (moderate)
    if engine == 'retrieval':
        return redirect(url_for('retrieve'))

    # 3) Generative / fallback chat
    if request.method == 'POST':
        text = request.get_json().get('message','').strip()
        if not text:
            return jsonify({'reply':''})

        # save user message
        db.session.add(Message(user_id=current_user.id,
                               sender='user', text=text))
        db.session.commit()

        # pick your engine
        # — generative branch —
    if engine == 'generative':
        return redirect(url_for('rule_chat'))
        # 
    # 4) GET: just render history (for generative or after suggestions)
    msgs = (Message.query
                    .filter_by(user_id=current_user.id)
                    .order_by(Message.timestamp)
                    .all())
    return render_template(
        'chat.html',
        messages=msgs,
        suggestion_flow=False,
        diagnosis=diagnosis,
        engine=engine
    )




#  Data for rule-based suggestions & follow-ups 
SUGGESTIONS = {
    "sad":      ["taking a short walk", "listening to uplifting music", "journaling your feelings"],
    "anxiety":  ["trying the 4–4–4 breathing exercise", "doing a 2-minute body-scan meditation", "grounding yourself by naming 5 things you see"],
    "stress":   ["taking a 5-minute stretch break", "doing a quick mindfulness break", "setting a small, achievable task"],
    "anger":    ["counting slowly from 10", "squeezing a stress ball safely", "visualizing a calm place while breathing"],
    "lonely":   ["reaching out to a friend", "joining an online support group", "writing a letter to someone you trust"],
    "insomnia": ["avoiding screens 30 minutes before bed", "sipping a warm caffeine-free tea", "trying progressive muscle relaxation"]
}
GENERAL_TIPS = ["taking a brief break", "trying a breathing exercise", "journaling your thoughts"]
FALLBACK_QUESTIONS = [
    "Could you tell me more about what’s on your mind?",
    "What do you think is contributing most to this feeling?",
    "How long have you been experiencing this?"
]
YES_RESPONSES = ["yes","sure","okay","please","yep","yeah"]
NO_RESPONSES  = ["no","not really","nah"]

# Helpers 
def next_suggestion(topic: str) -> str:
    """Return a tip for `topic` without repeating until exhausted."""
    pool = SUGGESTIONS.get(topic, GENERAL_TIPS)
    used_key = f"used_{topic}"
    used = session.get(used_key, [])
    choices = [tip for tip in pool if tip not in used]
    if not choices:
        used = []
        choices = pool.copy()
    pick = random.choice(choices)
    used.append(pick)
    session[used_key] = used
    return pick

def rule_based_reply(user_text: str) -> str:
    """Generate an empathetic rule-based reply, with yes/no follow-up support."""
    txt = user_text.lower()

    # 1) If we asked a follow-up, handle it
    if session.get("awaiting_followup"):
        session["awaiting_followup"] = False
        topic = session.get("last_topic", "general")
        if any(w in txt for w in YES_RESPONSES):
            session["awaiting_followup"] = True
            tip = next_suggestion(topic)
            return f"Sure—another idea for {topic} is {tip}. Would you like one more tip?"
        if any(w in txt for w in NO_RESPONSES):
            return "Okay, no problem. If there’s anything else you’d like to share, I’m here."
        # treat anything else as elaboration
        session["awaiting_followup"] = True
        tip = next_suggestion(topic)
        return f"Thank you for sharing more. You might also try {tip}. Does that help?"

    # 2) Detect new topic keyword
    for topic in SUGGESTIONS:
        if topic in txt:
            session["last_topic"] = topic
            session["awaiting_followup"] = True
            tip = next_suggestion(topic)
            q   = random.choice(FALLBACK_QUESTIONS)
            return f"I’m sorry you’re experiencing {topic}. One idea is {tip}. {q}"

    # 3) Catch-all
    session["last_topic"] = "general"
    session["awaiting_followup"] = True
    tip = next_suggestion("general")
    q   = random.choice(FALLBACK_QUESTIONS)
    return f"Thank you for sharing. Sometimes {tip} can help. {q}"

def do_triage_and_signoff(user_text: str, turns: int) -> str:
    """Generate the final recommendation based on the questionnaire."""
    cond_key  = session.get("condition", "Self-Care Recommended")
    diagnosis = session.get("diagnosis", "").lower()
    # Map to a human-friendly phrase
    ISSUE_MAP = {
        "Self-Care Recommended": "some extra self-care strategies",
        "Depression":            "symptoms of depression",
        "Anxiety":               "anxiety",
        "OCD":                   "obsessive-compulsive tendencies",
        "PTSD":                  "post-traumatic stress symptoms"
    }
    issue = ISSUE_MAP.get(cond_key, cond_key)
    msg   = f"Based on the questionnaire, it looks like you’re experiencing {issue}"
    if diagnosis:
        msg += f" ({diagnosis})."
    else:
        msg += "."
    msg += (
        " It could be really helpful to consult a licensed therapist "
        "who can give you personalized support. I’m going to end our session now—take care of yourself."
    )
    return msg

#  Chat endpoint
@app.route("/generative", methods=["GET","POST"])
@login_required
def rule_chat():
    # ensure the user went through the questionnaire
    if session.get("engine") != "generative":
        return redirect(url_for("questionnaire"))

    # on GET, reset turn count & show UI
    if request.method == "GET":
        session["turns"] = 0
        return render_template("generative.html")

    data = request.get_json(force=True) or {}
    msg  = data.get("message","").strip()
    if not msg:
        return jsonify(reply="", turns=session.get("turns",0), done=False)

    # increment turn counter
    turns = session.get("turns",0) + 1
    session["turns"] = turns

    # first 4 turns → rule-based empathy
    if turns <= 4:
        reply = rule_based_reply(msg)
        done  = False
    # 5th turn+ → final sign-off using questionnaire
    else:
        reply = do_triage_and_signoff(msg, turns)
        done  = True

    return jsonify(reply=reply, turns=turns, done=done)




#  Retrieval-only endpoint
#  Disorder-specific retrieval (moderate symptoms) 
#  MCQ‐Mapping Retrieval Flow 
#  MCQ‐Mapping Retrieval Flow 
RETRIEVAL_QS = [
  {'id':'q1','text':"When do these symptoms bother you most?",    'options':["Morning","Afternoon","Evening","Night"]},
  {'id':'q2','text':"Which activity makes you feel worse?",       'options':["Work/School","Social events","Being alone","Physical tasks"]},
  {'id':'q3','text':"How often do you have these thoughts/feelings per day?", 'options':["1–2 times","3–5 times","6–10 times","More than 10 times"]},
  {'id':'q4','text':"Which coping strategy have you tried already?", 'options':["Talking to a friend","Meditation/breathing","Exercise","Nothing yet"]},
  {'id':'q5','text':"How long have you been noticing these symptoms?", 'options':["< 2 weeks","2–4 weeks","> 4 weeks","Several months"]}
]

MAPPING = {}
for cond in ('depression','anxiety','ocd','ptsd','selfcarerecommended'):
    for q in RETRIEVAL_QS:
        for idx, _ in enumerate(q['options']):
            MAPPING[(cond, q['id'], idx)] = f"suggestion_{cond}_{q['id']}_{idx}"

@app.route('/retrieve', methods=['GET','POST'])
@login_required
def retrieve():
    if session.get('engine')!='retrieval':
        return redirect(url_for('chat'))

    if request.method=='GET':
        session['ret_step']    = 0
        session['ret_answers'] = []
        q0 = RETRIEVAL_QS[0]
        return render_template('retrieval.html', question=q0['text'], options=q0['options'])

    # record the user’s choice
    data   = request.get_json(silent=True) or {}
    choice = int(data.get('choice', 0))
    session['ret_answers'].append(choice)
    session['ret_step'] += 1
    step = session['ret_step']

    if step < len(RETRIEVAL_QS):
        qn = RETRIEVAL_QS[step]
        return jsonify({'type':'question','question':qn['text'],'options':qn['options']})

    # Build MCQ‐specific tips
    cond = session.get('condition','Self-Care Recommended')
    # normalize: remove spaces/hyphens, lowercase
    cond_key = ''.join(ch for ch in cond if ch.isalnum()).lower()

    tips = []
    for q, idx in zip(RETRIEVAL_QS, session['ret_answers']):
        tag = MAPPING.get((cond_key, q['id'], idx))
        tips += retr_responses.get(tag, [])

    # Fallback to generic if no MCQ‐specific tips found
    if not tips:
        generic_tag = f"suggestion_{cond_key}"
        tips = retr_responses.get(generic_tag, [])
        # if still empty, do case‐insensitive lookup
        if not tips:
            for t, resp in retr_responses.items():
                if t.lower() == generic_tag:
                    tips = resp
                    break

    session.pop('ret_step')
    session.pop('ret_answers')

    return jsonify({'type':'final','condition':cond,'tips':tips})


# map diagnosis labels to numbers for plotting
DIAG_SEVERITY = {'None':0,'Mild':1,'Moderate':2,'Severe':3}


@app.route('/api/results')
@login_required
def api_results():
    # DAILY average total‐score
    daily = (
        db.session.query(
            func.date(Result.timestamp).label("period"),
            func.avg(Result.score).label("value")
        )
        .filter_by(user_id=current_user.id)
        .group_by(func.date(Result.timestamp))
        .order_by(func.date(Result.timestamp))
        .all()
    )

    # WEEKLY average total‐score (ISO week)
    weekly = (
        db.session.query(
            func.strftime("%Y-%W", Result.timestamp).label("period"),
            func.avg(Result.score).label("value")
        )
        .filter_by(user_id=current_user.id)
        .group_by(func.strftime("%Y-%W", Result.timestamp))
        .order_by(func.strftime("%Y-%W", Result.timestamp))
        .all()
    )

    def serialize(rows):
        return [{"period": r.period, "value": round(r.value, 1)} for r in rows]

    return jsonify({
      "daily":  serialize(daily),
      "weekly": serialize(weekly)
    })

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')





# Hard-coded API token 
HF_TOKEN = "YOUR HUGGING FACE API KEY"
@app.route("/cohere_chat", methods=["GET", "POST"])
def cohere_chat():
    if request.method == "GET":
        return render_template("cohere_chat.html")

    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Whenever you're ready, I'm here to listen. 🤗"}), 400

    # Enhanced prompt with empathy guidelines
    prompt = f"""You are an empathetic mental health supporter. Follow these guidelines:
    1. Acknowledge the user's feelings first
    2. Provide a brief, supportive response
    3. Ask an open-ended question to continue the conversation
    4. Keep responses under 2 sentences
    
    Examples:
    User: I'm feeling really stressed about work
    Counselor: I hear how stressed work is making you feel. Would you like to share what specifically is troubling you?
    
    User: I can't stop worrying about everything
    Counselor: That sounds really overwhelming. Have you noticed any particular times when the worrying gets worse?
    
    User: {user_msg}
    Counselor:"""

    try:
        resp = requests.post(
            "YOUR_HUGGING_FACE_SECRET_API_ENDPOINT",  # Replace with your actual Hugging Face API endpoint
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
            timeout=10
        )
        
        if resp.status_code != 200:
            raise Exception(f"API error: {resp.status_code}")
            
        # Extract and clean the response
        reply = resp.json().get("generated_text", "").strip()
        
        # Post-processing to ensure quality
        if not reply or "I don't know" in reply or "I don't understand" in reply:
            raise Exception("Empty or unclear response from API")
            
        # Ensure the response is from the counselor perspective
        if "User:" in reply:
            reply = reply.split("Counselor:")[-1].strip()
            
        # Fallback to rule-based if response is too short
        if len(reply.split()) < 3:
            raise Exception("Response too short")
            
        return jsonify({"reply": reply})
        
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        # Enhanced fallback system
        try:
            # Try to use the few-shot template first
            few_shot_reply = few_shot.format(user_text=user_msg).split("Bot:")[-1].strip()
            if few_shot_reply:
                return jsonify({"reply": few_shot_reply})
                
            # Fall back to rule-based response
            rule_based = rule_response(user_msg)
            return jsonify({"reply": rule_based if rule_based else 
                           "I'm here to listen. Could you tell me more about how you're feeling?"})
            
        except Exception as fallback_error:
            app.logger.error(f"Fallback error: {str(fallback_error)}")
            return jsonify({"reply": "I'm listening. Please share what's on your mind."})


with app.app_context():
     db.create_all()

if __name__=='__main__':
    app.run(debug=True, use_reloader=False)
