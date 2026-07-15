# generate_retrieval_intents.py

import json
from pathlib import Path

# 1) Generic and fallback intents
generic_intents = [
    {
      "tag": "suggestion_greeting",
      "patterns": [],
      "responses": [
        "Thanks for completing the questionnaire! Based on your responses, here are some gentle suggestions you might find helpful:"
      ],
      "context": [""]
    },
    {
      "tag": "suggestion_depression",
      "patterns": [],
      "responses": [
        "Take short daily walks in nature to lift your mood.",
        "Try writing down three things you’re grateful for each evening.",
        "Maintain a regular sleep schedule—even small changes help.",
        "Reach out to a friend or family member for a quick chat."
      ],
      "context": [""]
    },
    {
      "tag": "suggestion_anxiety",
      "patterns": [],
      "responses": [
        "Practice 5 minutes of deep belly breathing exercises every morning.",
        "Schedule short breaks during your day to stretch or meditate.",
        "Keep a ‘worry journal’—write down anxious thoughts and let them go.",
        "Try a guided relaxation or mindfulness app before bed."
      ],
      "context": [""]
    },
    {
      "tag": "suggestion_OCD",
      "patterns": [],
      "responses": [
        "Use a timer for checking rituals—limit each check to 2 minutes.",
        "Challenge intrusive thoughts by naming and deferring them.",
        "Engage in a simple hobby (e.g., coloring, puzzles) to redirect focus.",
        "Create a small daily routine chart and stick to it rigidly."
      ],
      "context": [""]
    },
    {
      "tag": "suggestion_PTSD",
      "patterns": [],
      "responses": [
        "Practice grounding—notice 5 things you can see, hear, and touch.",
        "Try a brief body scan meditation before sleep.",
        "Connect with a trauma-informed support group online.",
        "Keep a small ‘comfort kit’ (photos, scent, music) handy."
      ],
      "context": [""]
    },
    {
      "tag": "suggestion_SelfCareRecommended",
      "patterns": [],
      "responses": [
        "You’re doing really well! Keep up your self-care routine.",
        "Spend 10 minutes doing something you love every day.",
        "Stay connected—plan a coffee or call with a friend this week.",
        "Remember to drink water and eat regular, balanced meals."
      ],
      "context": [""]
    },
    {
      "tag": "followups_retrieval",
      "patterns": [],
      "responses": [
        "How long have you been experiencing these feelings?",
        "What kinds of things tend to trigger these feelings for you?",
        "Have you talked to anyone else—friends or family—about how you’re feeling?",
        "On a scale of 1–10, how intense would you say these feelings are?"
      ],
      "context": [""]
    }
]

# 2) MCQ-specific tips mapping: two tips per option
tips_mapping = {
  'depression': {
    'q1': {
      0: ["Try morning sunlight exposure for 10 minutes.","Write down three positive thoughts in the morning."],
      1: ["Take a midday walk to refresh.","Listen to uplifting music in the afternoon."],
      2: ["Engage in relaxing evening activities like reading.","Practice gentle yoga stretches after dinner."],
      3: ["Limit caffeine intake in the evening.","Try a calming bedtime routine with warm tea."]
    },
    'q2': {
      0: ["Break tasks into smaller steps to reduce overwhelm.","Schedule short breaks during work to rest."],
      1: ["Engage in smaller social settings or one-on-one chats.","Set a time limit for social events and pause when needed."],
      2: ["Plan a comforting solo activity like reading.","Set up a relaxing space at home to unwind."],
      3: ["Incorporate light exercises like walking.","Try a brief indoor yoga session to boost mood."]
    },
    'q3': {
      0: ["Acknowledge these thoughts and gently shift focus.","Practice a 2-minute grounding exercise each time."],
      1: ["Use a worry journal to jot down thoughts.","Take short mindfulness breaks after each occurrence."],
      2: ["Practice a 5-minute breathing exercise every time.","Consider talking to a friend when frequency increases."],
      3: ["Seek professional support if thoughts spike.","Try a guided meditation app for intense moments."]
    },
    'q4': {
      0: ["Continue talking to trusted friends and share.","Set up regular check-ins with someone supportive."],
      1: ["Try different meditation styles like guided imagery.","Use a meditation app for structured breathing."],
      2: ["Combine exercise with social support—group walks.","Set achievable exercise goals to build confidence."],
      3: ["Start simple self-care tasks like a warm bath.","Explore relaxation techniques like calm music."]
    },
    'q5': {
      0: ["Often new symptoms pass—practice daily self-care.","Monitor changes and note improvements."],
      1: ["Try a new coping strategy like journaling.","Maintain a consistent sleep schedule this week."],
      2: ["Consider reaching out to a professional.","Continue self-care routines and note triggers."],
      3: ["Join a support group for encouragement.","Develop a structured routine to manage symptoms."]
    }
  },
  'anxiety': {
    'q1': {
      0: ["Practice 5 minutes of deep breathing upon waking.","Start your day with a calming stretch routine."],
      1: ["Take a mini mindfulness break mid-day.","Sip herbal tea to calm afternoon nerves."],
      2: ["Engage in relaxing activities like reading.","Use guided relaxation audio before bed."],
      3: ["Write down worries in a journal.","Practice progressive muscle relaxation at night."]
    },
    'q2': {
      0: ["Take short desk breaks to breathe deeply.","Organize tasks to reduce overwhelm."],
      1: ["Plan small social gatherings for comfort.","Use grounding techniques if you feel overwhelmed."],
      2: ["Practice self-soothing activities like coloring.","Reach out to a friend if anxiety spikes."],
      3: ["Break tasks into smaller steps.","Play soothing music while doing physical tasks."]
    },
    'q3': {
      0: ["Note each occurrence and challenge anxious thoughts.","Practice a brief grounding exercise each time."],
      1: ["Keep a worry log to record anxious thoughts.","Take a 2-minute breathing break after each episode."],
      2: ["Try a guided anxiety-management app.","Use a calming mantra to steady your mind."],
      3: ["Consider talking to a counselor.","Practice a 10-minute meditation to lower intensity."]
    },
    'q4': {
      0: ["Continue sharing with supportive friends.","Use role-play to rehearse difficult conversations."],
      1: ["Experiment with different breathing exercises.","Follow a structured meditation routine daily."],
      2: ["Incorporate gentle exercise like walking.","Try yoga poses specifically for anxiety relief."],
      3: ["Start practicing one coping strategy today.","Explore art or music therapy techniques."]
    },
    'q5': {
      0: ["Monitor symptoms and practice coping daily.","Notice patterns in your triggers."],
      1: ["Increase self-care efforts like regular exercise.","Schedule a check-in with a therapist."],
      2: ["Seek professional support if anxiety persists.","Maintain a regular anxiety-management routine."],
      3: ["Join an anxiety support group.","Discuss long-term strategies with a counselor."]
    }
  },
  'ocd': {
    'q1': {
      0: ["Use a 2-minute timer for morning checks.","Practice delaying one urge each time."],
      1: ["Redirect focus to a hobby mid-day.","Set specific times for checking behaviors."],
      2: ["Engage in a calming activity like reading.","Use exposure exercises to challenge rituals."],
      3: ["Journal progress before bed.","Practice relaxation to ease nighttime rituals."]
    },
    'q2': {
      0: ["Limit checking your workspace to set periods.","Use alarms to end checks."],
      1: ["Challenge intrusive thoughts with affirmations.","Plan small distractions from rituals."],
      2: ["Engage in a hobby to redirect focus.","Schedule brief tasks to break compulsions."],
      3: ["Allow one check per task then move on.","Use deep breathing when urges arise."]
    },
    'q3': {
      0: ["Pause for 3 seconds before responding to urges.","Use a grounding object like a stress ball."],
      1: ["Label intrusive thoughts and let them pass.","Practice thought-defusion techniques."],
      2: ["Challenge compulsion beliefs gently.","Use a mantra like ‘thoughts are not facts.’"],
      3: ["Seek professional guidance if urges intensify.","Use a CBT worksheet to track compulsions."]
    },
    'q4': {
      0: ["Continue sharing with trusted people.","Use their support to resist urges."],
      1: ["Follow a guided breathing script for OCD.","Combine breathing with mindfulness."],
      2: ["Incorporate physical activity to disrupt rituals.","Take a short walk when urges rise."],
      3: ["Start with one simple coping strategy.","Explore professional self-help resources."]
    },
    'q5': {
      0: ["Monitor patterns in your compulsions.","Keep a brief log of urges."],
      1: ["Set small goals to reduce checks.","Reward yourself for resisting urges."],
      2: ["Consider consulting an OCD specialist.","Implement a daily exposure hierarchy."],
      3: ["Join an OCD support group.","Discuss ERP strategies with a professional."]
    }
  },
  'ptsd': {
    'q1': {
      0: ["Practice grounding after waking.","Read a comforting passage from a favorite book."],
      1: ["Use a grounding object to stay present.","Take a nature walk to reduce flashbacks."],
      2: ["Try a soothing body-scan meditation.","Journal any distressing memories."],
      3: ["Create a safe bedtime routine.","Use calming music before sleep."]
    },
    'q2': {
      0: ["Set a private decompression space at work.","Use mindfulness breaks after stress."],
      1: ["Attend events with a trusted friend.","Use grounding if triggered."],
      2: ["Reach out to a support hotline.","Keep comforting items close by."],
      3: ["Incorporate grounding into tasks.","Listen to calming music while working."]
    },
    'q3': {
      0: ["Note flashbacks and remind yourself you’re safe.","Practice the 5-4-3-2-1 grounding technique."],
      1: ["Use a brief breathing exercise afterward.","Jot down triggers for therapy discussion."],
      2: ["Practice guided imagery exercises.","Use a mantra like ‘I am safe now.’"],
      3: ["Seek immediate support if flashbacks spike.","Contact your support network."]
    },
    'q4': {
      0: ["Discuss experiences with a trauma-informed friend.","Use storytelling to process memories."],
      1: ["Follow a guided body-scan meditation daily.","Focus on breathing to reduce tension."],
      2: ["Try gentle exercise to ease stress.","Engage in grounding through movement."],
      3: ["Start with one relaxation strategy.","Explore therapy-based self-help tools."]
    },
    'q5': {
      0: ["Practice grounding daily.","Keep a self-care kit nearby."],
      1: ["Incorporate professional check-ins.","Develop a consistent coping routine."],
      2: ["Schedule sessions with a trauma therapist.","Use journaling to process memories."],
      3: ["Join a PTSD support group.","Work on exposure techniques with a therapist."]
    }
  }
}

# 3) Assemble all intents
all_intents = generic_intents.copy()
for domain, qmap in tips_mapping.items():
    for qid, opts in qmap.items():
        for idx, tips in opts.items():
            all_intents.append({
                "tag": f"suggestion_{domain}_{qid}_{idx}",
                "patterns": [],
                "responses": tips,
                "context": [""]
            })

# 4) Write out retrieval.json
out = {"intents": all_intents}
dest = Path(__file__).parent / "Models" / "retrieval.json"
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("✅ Rebuilt Models/retrieval.json with disorder-specific MCQ tips.")
