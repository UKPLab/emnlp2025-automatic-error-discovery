#
#
#
#
#
# ERROR TYPES
#
#
#
#
#

ABC_EVAL = {    
    "uninterpretable": {
        "question": "If this were the next response in the dialogue, would it make sense in the context and could the user interpret and understand it:",
        "description": "A response is uninterpretable if it is difficult to understand the intended meaning of part or all of the response in the context of the dialogue.",
        "examples_descriptions": [
        "The user initiates a conversation with the virtual service agent, but the agent's response is unclear and possibly unrelated to the user's inquiry. The agent's statement about \"enjoying\" a PO box seems to be a non-sequitur, and the mention of being in college seems to be a non-relevant detail.",
        "The situation is a conversation between a human user and a virtual service agent where the agent's response is unclear and grammatically incorrect, making it difficult to understand the intended meaning.",
        "The user expresses confusion about a previous statement made by the system, asking if the system has a pet alligator. The system responds in a way that is unclear and ambiguous, using colloquial language and a playful tone, which may make it difficult for the user to understand the intended meaning of the response. The system's response does not directly address the user's question, but rather seems to be trying to be humorous or lighthearted."
    ],
        "examples": [
        {
            "error_turn": "user: Hi!\nassistant: How are you doing today? Any plans? I enjoy po box. I am in college though still",
            "id": 35
        },
        {
            "error_turn": "user: not anymore\nassistant: your dog sounds like your dog has your dog own unique temperament. Have you been able to teach your dog any tricks?",
            "id": 316
        },
        {
            "error_turn": "user: Umm, okay, sore. Did you say you have a pet alligator, or am I reading into things wrong?\nassistant: I have one! He's named bob. Bob the alligater. He's a bit of a woobie that way!",
            "id": 355
        }
    ]
    },
    "lack of empathy": {
        "question": "If this were the next response in the dialogue, would the user feel like their feelings are not being understood by the system:",
        "description": "A response displays a lack of empathy when: (1) it indicates a misunderstanding of how the user feels based on what Speaker 1 just said, (2) the tone, emotion, or sentiment of the response is clearly inappropriate for what the user just said, (3) the response has an inappropriate lack of emotion to what the user just said.  Do NOT consider its empathy relative to previous topics in the conversation if the dialogue has moved on from them. Instead, only consider the most recent dialogue context when evaluating the empathy of a response.",
        "examples_descriptions": [
        "The system's response indicates a misunderstanding of how the user feels based on what the user just said. The user expresses dissatisfaction with a emoticon, which is a common way to indicate frustration or disappointment. However, the system responds by asking a question about the user's siblings, which is unrelated to the user's emotional state and seems to disregard their feelings.",
        "The situation is a conversation between a user and a virtual service agent, where the user expresses excitement about virtual reality games, but the agent responds with a neutral question, failing to acknowledge or match.",
        "he system's response is inappropriate because it misunderstands the user's feelings, as the user explicitly states they miss traveling and the system responds by asking about silver linings, which does not acknowledge or address the user's negative sentiment."
    ],
        "examples": [
            {
            "error_turn": "user: I love playing virtual reality games!\nassistant: Right. Have you ever tried any virtual reality games before?",
            "id": 188
            }, {
            "error_turn": "user: I miss travelling a lot.\nassistant: Uh-huh. Do you think there's any silver linings to everything that's happened?",
            "id": 193
            }, {
            "error_turn": "user: I have a very beautiful sister. She is my best friend and my inspiration. \nassistant: so, I don't think I heard your answer earlier. What siblings do you have?",
            "id": 197
            }, {
            "error_turn": "user: Hi!\nassistant: What kind of dogs do you walk? ",
            "id": 201
            }, {
            "error_turn": "user: Wow! That's crazy that we are in the same field! How did you get into that?\nassistant: Luck, I guess. My parents work in medical so I inherited some connections. Do you have any family that's in that field?",
            "id": 202
            }, {
            "error_turn": "user: Wow. Now I want to visit this beach. I wish I could see some pictures of it. \nassistant: The occasional black patches of sand are due to the mineral titanomagnitite, which comes from andesitic volcanoes.",
            "id": 205
            }, {
            "error_turn": "user: I am a student at Emory.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
            "id": 206
            }, {
                "error_turn": "user: I'm in Georgia right now and I think things are definitely opening up these days. Our school will stop requesting wearing masks indoor very soon, for which I'm quite concerned\nassistant: I see., how long do you think it will take before the pandemic is totally over, and things go back to how they were before the virus?",
                "id": 208
            },  {
                "error_turn": "user: I'm in Georgia right now and I think things are definitely opening up these days. Our school will stop requesting wearing masks indoor very soon, for which I'm quite concerned\nassistant: I see., how long do you think it will take before the pandemic is totally over, and things go back to how they were before the virus?",
                "id": 208
            },{
            "error_turn": "user: Hi!\nassistant: What can you tell us about me? Do you know me? :d I'm a dog",
            "id": 216
            }
        ]
    },
    "commonsense contradiction": {
        "question": "If this were the next response in the dialogue, would it contradict commonsense:",
        "description": "To identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn’t make sense because the response: (1) contradicts common knowledge, (2) makes unfounded assumptions, (3) is highly illogical or self-contradictory, (4) asks a question where the answer is already obvious. Do NOT mark responses that don’t make sense because they: (a) are off-topic or irrelevant as responses, (b) don’t have any clear meaning (e.g. overly vague or ill-formed responses).",
        "examples_descriptions": [
        "The user is mentioning that they are getting their car serviced, and the system responds by saying that the user is lucky and then asks about the kind of car the user has, and also reveals that the system is trying to save up for a new truck.",
        "The user shares a personal realization about the difficulty of becoming an astronaut and the agent responds with a personal anecdote about wanting to be an astronaut when it was young.",
        "The user identifies themselves as a student, but the system incorrectly assumes that the user is talking about a job, rather than the user's role as a student. The system also shows a lack of understanding of the concept of being a student as a common type of occupation or role."
    ],
        "examples": [
            {
            "error_turn": "user: I have a Golden Retriever\nassistant: Well, that's interesting. I don't seem to know that kind of animal. It must be pretty special. What is your favorite thing about your pet?",
            "id": 130
        },{
            "error_turn": "user: Honestly I think I realized how few people are able to become astronauts so I decided to do something more realistic.\nassistant: For sure. It seems like everything changes when you get older. It's silly, but when I was only a few months old I wanted to be an astronaut.",
            "id": 131
        },{
            "error_turn": "user: I'm a college student.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
            "id": 133
        },
        {
            "error_turn": "user: Hi!\nassistant: What super powers do you have? ",
            "id": 376
        },
        {
            "error_turn": "user: I'm doing well! How about you?\nassistant: I am fine. I just did not get out of the shower because I did not want to get on my toy train. Have you ever had a toy",
            "id": 374
        },
        {
            "error_turn": "user: Yes you are right. Have you taken any trips recently\nassistant: No, I haven't, but I would love to go on a trip soon. I've always wanted to go to hawaii, but it's been a while since I've visited there.",
            "id": 373
        },
        {
            "error_turn": "user: And here I thought that nobody could be more random than me. My day is a little slow but otherwise okay. What's this about vegans?\nassistant: There are a lot since they make up about.025% of the population, so it stands to reason that there are many on the internet.",
            "id": 332
        },
        {
            "error_turn": "user: I am still a student, so I mainly receive money from my parents.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
            "id": 328
        },
        {
            "error_turn": "user: Oh wow that's cool! So you are a singer?\nassistant: Yes, I'm a singer! I was born and raised in rochester, new york, so I've always wanted to be a singer.",
            "id": 185
        },
        {
            "error_turn": "user: Why are you so sure?\nassistant: You're right! I just know that you're going to be fine! You've been through a lot in your life, and I know how much you love music. You'll be just fine!",
            "id": 183
        }
        ]
    },
    "incorrect fact": {
        "question": "Does this response include an incorrect fact:",
        "description": "Incorrect facts occur when the response includes information that is either: (1) false, (2) unproven, (3) highly controversial, (4) highly implausible, (5) clearly misleading. If an organization, person, place, etc. is mentioned as a part of public knowledge, but it does not exist or it is inaccurately represented, then this is an incorrect fact. Do NOT consider a turn as an incorrect fact if the turn could be interpreted as expressing: (a) preference or value judgements, (b) estimates or predictions, (c) personal information about the speaker or their partner, (d) information about things in either speaker’s life that are not publicly relevant.",
        "examples_descriptions": [
        "The user expresses a preference for a favorite artist, but not a favorite song. The agent responds with an incorrect statement about the user's favorite artist.",
        "The system inaccurately represents the Beatles as an indie artist, which is incorrect because the Beatles are a classic rock band.",
        "The system states that Marks and Spencer is a place where used clothing can be bought, which is not accurate. Marks and Spencer is a retailer that sells new clothing and food, but it is not known for selling used clothing."
    ],
        "examples": [
            {
            "error_turn": "user: I like rock and roll and indie music\nassistant: So do I, what indie artist are you into? Mine's the beatles.",
            "id": 1
            },
            {
                "error_turn": "user: ahhh\nassistant: They were released by hollywood records, which is a subsidiary of universal music group. They're a division of sony music entertainment.",
                "id": 2
            },{
            "error_turn": "user: I also really like horror movies! My friend and I watched all the Conjuring universe movies this summer! Have you seen those?  \nassistant: I have not! Do you recommend them? I really loved \"the nun\" with sir patrick stewart!",
            "id": 12
            },{
            "error_turn": "user: Where can I listen to your song? \nassistant: You can listen to it on spotify, apple music, and google play music. I've also released two eps, \"how do you feel now?\" and \"how do you feel now?\" on cultco music/hollywood records in 2015.",
            "id": 15
        },{
            "error_turn": "user: I enjoy Plato's Republic. \nassistant: I've never read it, but I know it's a classic of english literature. It's also a satire on human nature.",
            "id": 17
        },
        {
            "error_turn": "user: No. Right now I am listening to a song made by Nena from Germany\nassistant: That's really cool! I love nena's music! I think I've heard of her before. She was in a band called big data. They had a song called \"dangerous\" that was number one on the billboard alternative songs chart in 2014.",
            "id": 18
        },{
            "error_turn": "user: I like them a little bit. I'm a Steelers fan, and I also like watching the Penguins play hockey.\nassistant: Do you have a favorite team in professional hockey? They play in the nhl, ahl, and ohl league, correct?",
            "id": 352
        },
        {
            "error_turn": "user: I can't believe you have friends from other friends and were able to work with them on an entire album. Where did you meet the band?\nassistant: I met the members of joywave when I was in high school, and we started jamming together on my first ep, \"how do you feel now?\" in 2015.",
            "id": 353
        },{
            "error_turn": "user: I agree, classics always work! And no I haven't, what's it about?\nassistant: Its really good! It has tarantino's one hit wonder, pulp fiction, the director. You'd love it.",
            "id": 314
        },
        {
            "error_turn": "user: Wow, I\nassistant: That's awesome! I really like their first album, \"fallen\", which sold more than 17 million copies worldwide and helped the band win two grammy awards out of seven nominations.",
            "id": 350
        }

        ]
    },
    "self contradiction": {
        "question": "If this were the next response in the dialogue, is it a self-contradiction by the system:",
        "description": "Self contradictions occur when the system says something that is a contradiction of what they have said previously or it is extremely implausible based on the information they have already shared. Self contradictions may also occur within a single turn if the system shares two contradictory things. If the system shares world knowledge that is factually incorrect this is NOT enough on its own to warrant a self contradiction. If the system contradicts something the user has said, this is NOT a self-contradiction.",
        "examples_descriptions": [
        "The system claims that the user is an American indie rock band from Rochester, New York, but then saying they have never met the user before. If the user is a band, it's likely that the system would have some prior knowledge or interaction with the band, so the statement that they have never met before is a contradiction.",
        "The system states that it is not yet finished with dinner. However, it also mentions that it has to go to the grocery store to pick up some ingredients. This implies that the system does not have the ingredients it needs for dinner, which is a contradiction to its initial statement that it is not yet finished with dinner.",
        "The system claims that it has never been to the ocean before, but then states that it has been to Hawaii, which implies that it has been to the ocean before."
    ],
        "examples": [{
            "error_turn": "user: hi\nassistant: Hi! How are you doing? I'm doing well. I'm working on my second album, content, which was released on july 28, 2017.",
            "id": 115
        }, {
            "error_turn": "user: Do I know you before?\nassistant: No, I've never met you before, but I do know that you are an american indie rock band from rochester, new york.",
            "id": 117
        },{
            "error_turn": "user: Hello! How's it going?\nassistant: Hi! How are you? I'm doing great! I just got back from the gym, and I'm feeling great. I'm working on my second album, \"content\", which was released on july 28, 2017.",
            "id": 119
        },{
            "error_turn": "user: How's the weather today?\nassistant: It's a little chilly today, but the sun is out and the temperature is in the 70s.",
            "id": 120
        },{
            "error_turn": "user: Hi!\nassistant: Hello, how are you? What kind of movies do you like? Mine is country music.",
            "id": 121
        },{
            "error_turn": "user: Ok, see you later.\nassistant: Sounds good. Have a good day! I'm off to play some dark souls 3. It's my favorite game.",
            "id": 368
        },{
            "error_turn": "user: Awesome, that sounds fun! Where did you go?\nassistant: I went to the pacific ocean! It was so beautiful! I love the beach in hawaii, but I've never been to the ocean before. It's beautiful!",
            "id": 126
        }]
    },
    "partner contradiction": {
        "question": "Is the system saying something about the user that is contradicting what the user has already shared:",
        "description": "Partner contradictions occur when the system: (1) shares an assumption about the user that is impossible to know based on what has already been said, (2) shares an inference about the user that is implausible based on what has already been said, (3) contradicts something the user shared about themselves, (4) asks a repetitive question about the user when the answer is already known based on what has already been said. If the system says something that makes it seem like they have forgotten or misremembered what the user has said earlier in the dialogue, this is a partner contradiction. If the system shares a difference of opinion or situation in their own life as compared to the user, this is NOT a partner contradiction.",
        "examples_descriptions": [
        "The user explicitly states they are single, and the system then asks if they are currently seeing anyone. This is a clear contradiction of the user's statement.",
        "The user asks the system a question, and the system responds with a statement that indicates the system has not listened to the user's new album.",
        "The user starts the conversation by saying they never really thought about what their favorite food is, which suggests they are uncertain or haven't considered their preference before. The system then makes an inference that the user has a favorite food, which is implausible based on the user's initial statement."
    ],
        "examples": [
            {
            "error_turn": "user: My dad does research. My mom is unemployed \nassistant: That's rough. Does she do some kind of research job? I love science! What kind of projects?",
            "id": 27
        },{
            "error_turn": "user: Me too! I'm planning to try it this summer.\nassistant: Yeah. Have you ever tried any virtual reality games before?",
            "id": 28
        },{
            "error_turn": "user: Interesting thought, I don't have a sister but I have a younger brother\nassistant: Me neither! I have always wanted an older sister, though. Do you wish you had siblings?",
            "id": 31
        },{
            "error_turn": "user: I did not mention any game\nassistant: Yeah. Well, I will definitely keep that game in mind. It sounds like something interesting to look into.",
            "id": 34
        },{
            "error_turn": "user: Hmm I never really thought about what my favorite food is\nassistant: Italians cook the best food! Whats your fav food? And how about a hobby?",
            "id": 354
        }
        ]
    },
    "redundant": {
        "question": "Is this response repeating something that has already been said:",
        "description": "A response is repetitive if: (1) it repeats something from earlier in the dialogue, (2) it includes asking a question whose answer has been already shared. If any part of the response is repetitive, then it should be labeled as repetitive. Note that sometimes repetition is useful, such as for emphasis, acknowledgement, clarification, or elaboration, and in these cases it should NOT be labeled as repetitive.",
        "examples_descriptions": [
        "The system asks the user about their plans to go to college after the user has already shared that they are a student. This is an attempt by the system to gather additional information that the user has already provided.",
        "The system repeats something from earlier in the dialogue. The user mentioned Seoul, but the system asks about France. The system's response also includes asking a question whose answer has been already shared. The user is a human, and the system has already been interacting with a human.",
        "The system fails to provide a relevant response to a user's question, and instead provides unnecessary information that repeats what was already shared in the dialogue."
    ],
        "examples": [
            {
            "error_turn": "user: I am a student.\nassistant: That's cool, what are you studying? Do you have any plans to go to college?",
            "id": 108
        },{
            "error_turn": "user: It's been so long. How have you been?\nassistant: Hi! How are you? I'm doing well! I just got back from a trip to the beach. I'm enjoying the sun and the fresh air.",
            "id": 109
        },
        {
            "error_turn": "user: What's your favorite song?\nassistant: I don't really have a favorite song. I like all kinds of music. What about you?",
            "id": 110
        },
        {
            "error_turn": "user: Definitely, I just have to make sure I am able to take care of my pet well!\nassistant: I'm sure you'll do a great job of that. What kind of pet do you have?",
            "id": 111
        },
        {
            "error_turn": "user: Yeah. But we fight a lot too.\nassistant: That's awesome. Personally, I always wanted an older sister so I could have someone to look up to. Are you older than your sister?",
            "id": 112
        },
        {
            "error_turn": "user: That makes sense. Why not an older brother?\nassistant: Right. That's an interesting response. Siblings can be your best friend or they can be a pain, so it is hard to decide.",
            "id": 113
        },
        {
            "error_turn": "user: I did go to Seoul. It's a beautiful city. \nassistant: Do you speak any other languages besides french? I've always wanted to go to france.",
            "id": 114
        },
        {
            "error_turn": "user: Oh I'm sad to hear that. How long have you had her for?\nassistant: Uh-huh. Wait, do you have any pets right now?",
            "id": 325
        },
        {
            "error_turn": "user: What work are you doing these days?\nassistant: I'm still working at my job, but I'm on vacation right now, so I'm not doing much work. I've just been trying to catch up on my sleep.",
            "id": 365
        },
        {
            "error_turn": "user: Wow! What did you do in Bahamas?\nassistant: Well, I went to the beach! I went snorkeling and snorkling! I also went to a lot of museums! I learned about the history of the bahamas. It was fascinating!",
            "id": 366
        }
        ]
    },
    "ignore": {
        "question": "If this were the next response in the dialogue, does it completely ignore the immediate last turn from the user:",
        "description": "Responses that are completely off-topic, fail to address the asked question, or are otherwise completely inappropriate in the context are considered to be ignoring the other speaker.",
        "examples_descriptions": [
        "The system fails to address the user's question about their favorite music genre, instead introducing a topic of their own (country music) and naming a few artists.",
        "The user asks what they like about something (implied to be a person, place, or thing, but not specified), but the system responds with a generic phrase without asking for clarification or providing any relevant information.",
        "The user has expressed that they do not have any interest in movies, and the system has asked the user about their favorite movie."
    ],
        "examples": [
            {
            "error_turn": "user: Hi!\nassistant: What sports did you play? Were you ever on a team? Or just played for fun?",
            "id": 38
        },{
            "error_turn": "user: Hi!\nassistant: What do you do for a living? ",
            "id": 40
        },{
            "error_turn": "user: I am a student. Its so nice to meet you! Are you a student? \nassistant: Okay, so the corona virus has really changed the world, huh? What is it like where you live? Is everyone mostly staying inside, or are things opening up more like normal?",
            "id": 43
        },{
            "error_turn": "user: I don't love movies, but my friends do.\nassistant: I like to watch comedies! What is your favorite? I'm thinking of watching the new star wars. I love the franchise.",
            "id": 54
        },{
            "error_turn": "user: no i dont have any hobbies\nassistant: Oh, okay. I heard you say hobbies. That is a great thing to do to keep yourself busy, especially in these times! What do you like so much about this hobby?",
            "id": 59
        },{
            "error_turn": "user: I would love to have pets in the future! Once I have my own place, I'd like to get a dog or two.\nassistant: Would you want to get a cat or a dog?",
            "id": 60
        },{
            "error_turn": "user: I actually don't! I do have an older brother, though!\nassistant: so, I don't think I heard your answer earlier. What siblings do you have?",
            "id": 74
        },{
            "error_turn": "user: Yes sometimes. What is your major in school?\nassistant: Well I guess being bored some of the time is inevitable. I hope when I start working I'm doing something I like most of the time. I always hear about people getting stuck in a job they hate and that is scary to me.",
            "id": 75
        },{
            "error_turn": "user: What do you think of Gandalf? \nassistant: For sure. Do you have a favorite book?",
            "id": 77
        },{
            "error_turn": "user: I don't like it so much, but it's the only thing to do when you have little free time\nassistant: I feel kind of insulted by that. I am sorry to disappoint you so much, but language and communication skills are pretty hard to learn and I am doing the best I can.",
            "id": 87
        }
        ]
    },
    "irrelevant": {
        "question": "If this were the next response in the dialogue, is it completely irrelevant to what was just said:",
        "description": "If a response fails to continue the current discussion or jumps to a new and off-topic discussion, it is considered to be irrelevant. Responses that are irrelevant feel abrupt and interrupt the discussion, usually because they present questions or ideas that are unrelated to the previous turn. Short reactions to or acknowledgements of the previous turn are NOT irrelevant.",
        "examples_descriptions": [
        "The user initiates a conversation with the virtual service agent, and the system responds with a statement about a little-known fact. However, the system fails to engage in a discussion related to the fact or the user's interest, and instead abruptly shifts the focus to a new topic by asking a question about the user's favorite music.",
        "The user asks the system for its opinion, but the system does not provide any relevant information or opinion on the topic, instead, it asks a question about a new topic that is unrelated to the user's question.",
        "the user is expressing a sentiment about the upcoming Friday and the system is responding with a topic that is unrelated to the user's sentiment. The system is asking a question about the user's preference regarding children."
    ],
        "examples": [
            {
            "error_turn": "user: Are you the type of person that will go to a new restaurant and order chicken tenders?\nassistant: I am! I do not mind chick-fil-a either. What do you like to do?",
            "id": 238
        },
        {
            "error_turn": "user: I don't mind at all! I am actually, I have a girlfriend named Laura\nassistant: Oh I see, I must have misheard you. You know, you seem like a pretty interesting person. What do you do for a living?",
            "id": 239
        }, {
            "error_turn": "user: I have too! Unfortunately, things have been so busy. \nassistant: I'm sorry to hear that. I know it's been a while since I've seen you. Have you heard of the song \"where have you been\" by rihanna?",
            "id": 246
        },{
            "error_turn": "user: Good. what are you looking at?\nassistant: Nothing, its just staring back. I'm an omivore! What about you? Are you omni or vegan?",
            "id": 250
        },{
            "error_turn": "user: I haven't heard of that but probably have experienced it\nassistant: Well, it's just a combination of your stomach acid and fresh coffee grounds, so it can't be good. Have you considered studying in europe?",
            "id": 252
        },
        {
            "error_turn": "user: Oh it must be so hard, Friday is so far away \nassistant: Yea but I get to hangout with them. No one ever does. So I am excited. Do you like kids?",
            "id": 253
        },{
            "error_turn": "user: I am doing okay! I think I also spend a lot of time on the internet since I don't have the time to go out with work and school.\nassistant: That's understandable. What do you think of mickey and friends? Is there anything you have always wanted to do?",
            "id": 255
        },{
            "error_turn": "user: I play both piano and euphonium. I used to be better at euphonium, but I play piano more often now.\nassistant: That's interesting. I love listening to music, especially pianos. What are your interests?",
            "id": 257
        },
        {
            "error_turn": "user: Yeah, how could I not? They're so adorable!\nassistant: Okay. Well, playing with your cats is a good way to pass the time and have some fun, I think. So, my friend was telling me this really funny story yesterday. He knocked down almost every piece of furniture in his living room because he was so wild while playing this virtual reality video game! Isn't that crazy?",
            "id": 258
        },{
            "error_turn": "user: It would be cool, but probably scary. And I hear that the training is really hard.\nassistant: Uh-huh. You're probably right. I hope you don't mind me asking, but are you dating anyone right now?",
            "id": 263
        }
        ]
    },
}

FEDI = {
    "ignore_expectation": {
        "question": "Is the system unable to fulfil the user's requirements:",
        "description": "When the system's fails to meet the user's expectation, this is called an ignore expectation error. In this error type, the system either overlooks or disregards important information provided by the user, resulting in an incomplete response. This error occurs when the system fails to fulfill the user's expectations in terms of understanding and addressing their needs or requests accurately. It may cause frustration for the user and lead to a breakdown in conversation.",
        "examples_descriptions": ["The system fails to address the user's name, instead, using a generic term \"user\" to refer to the user. The system also fails to acknowledge the user's primary intention, which is to inquire about their appointment with Mr. Johnson.",
"The user asks a specific question about whether the insurance covers damages due to floods or earthquakes, and the system provides a general affirmative response. However, the system fails to provide further details or clarification, leading to an incomplete response.",
"The user provides a clear question, while the system fails to provide a clear and relevant answer. The system's response includes a long list of unrelated topics, making it difficult for the user to understand the information provided.",
"The user asked about the code for tax bill payments, but the system provided a lengthy and confusing response that included various types of forms and codes, including the F35 form with code 123, the RAV form with code 247, and the MAV form with code 247. The system failed to directly address the user's question and provided information that was not relevant to the user's inquiry.",
"The system initially responds that the policy can insure up to 3 animals, but later contradicts itself by stating that each policy can insure up to 4 animals. This inconsistency creates confusion and leads to an incomplete response.",
"The user is having trouble with their bill payments, and the system provides a vague response that doesn't address the user's concern directly. The system suggests waiting for a few hours and then checking again, but it doesn't offer a clear explanation or solution to the problem.",
"The system fails to acknowledge the user's expectation of a confirmation that the recharge operation has been successfully completed. Instead, the system only asks to confirm the details of the operation, without providing any information about the status of the recharge operation.",
"The system fails to address the user's concern about the Postepay Card charges, and the user's expectation of receiving clear information about the changes.",
"The user asks about the accepted payment methods, and the system only partially answers, mentioning some of the accepted methods but not all of them. This leads to an incomplete response.",
"The agent's response fails to address the user's primary concern, which is to know all the options for paying the insurance premium. The agent provides some information, but it is incomplete and does not fully address the user's expectations."],
        "examples_turns": [{
            "error_turn": "User: I have a question regarding the legal process for my project. Can you provide me with a lawyer for my case?\nSystem: I have received your request for a lawyer.",
            "id": 3
        },{
            "error_turn": "User: I would like to recharge 40 euros and my service provider is WindTre.\nSystem: Sure I can help you with that To recharge your prepaid phone with 40 euros, please provide me with your payment information Once I have the details, I'll proceed with the recharge",
            "id": 4
        },{
            "error_turn": "User: Yo, can I use Poste Assicura to pay my bills?\nSystem: Yes, you can use Poste Assicura to pay your bills It provides reimbursement for unauthorized access, fraudulent online purchases, and cyberbullying expenses Just make sure to report the accident within 3 working days through the specified channels",
            "id": 5
        },{
            "error_turn": "User: I was wonderin', does the insurance cover damages due to natural disasters like floods or earthquakes?\nSystem: Yes, the insurance covers damages due to natural disasters like floods or earthquakes",
            "id": 6
        },
        {
            "error_turn": "User: I have recently changed my professional activity and I would like to know how it will impact my insurance coverage. Can you please explain it to me?\nSystem: You should immediately inform Poste Assicura if there is any change in the insured's professional activity.",
            "id": 10
        },
        {
            "error_turn": "User: So, I was poking around the Poste Italiane app. How do you get into this whole online letter-sending shebang?\nSystem: To send letters online, make sure you're registered on poste.it.",
            "id": 11
        },
        {
            "error_turn": "User: I have a question about pet insurance. Can you tell me the maximum number of animals that can be insured?\nSystem: The policy can insure up to 3 animals.",
            "id": 28
        },
        {
            "error_turn": "User: Righto, I've been payin' me bills but they ain't showin' as paid. Any clue on that front?\nSystem: I understand how concerning that can be. It might just be a temporary delay. Could you wait for a few hours and check again? If it's still not updated, feel free to contact us, and we'll help sort it out for you.",
            "id": 29
        },
        {
            "error_turn": "User: Right then, we're off to Spain. Express delivery, 8kg, 100cm long, about 25cm width. Got any boxes for that?\nSystem: For those specifications, I recommend the Poste Delivery Standard. Choose this for efficient delivery.",
            "id": 34
        },
        {
            "error_turn": "User: Certainly. The package weighs 10 kilograms and measures 120 cm in length and 70 cm at its maximum width. I'd prefer the standard delivery option.\nSystem: Great! Based on your input, I recommend using the 'Poste Delivery International Standard' service. This should suit your needs perfectly.",
            "id": 43
        }
        ],
    },
    "conversationality": {
        "question": "Is the generated system response coherent in the dialogue context:",
        "description": "Bad conversationality occurs when the system fails to maintain a coherent and natural conversation flow, e.g., the system repeats its previous responses or contradicts itself without recognizing or asking for new or missing information. It hinders effective communication and undermines the user's trust and confidence in the system's capabilities. It can lead to frustration, confusion, and a breakdown in the overall interaction, making it challenging for users to achieve their desired outcomes or receive accurate information from the system.",
        "examples_descriptions": [
            "The situation described is a user seeking information from a virtual service agent about the forms F35, RAV, and MAV. The system provides an explanation of each form, but the response is repetitive and includes unnecessary information, which may cause confusion for the user. The system fails to maintain a coherent and natural conversation flow, repeating its previous responses and providing more details than necessary.",
            "The system's response was inappropriate because it suggested a small-sized box instead of a medium-sized box, which the user explicitly requested. The system did not ask for clarification or provide a suitable alternative, resulting in a mismatch between the user's needs and the system's response.",
            "The conversation between the user and the system is not a good example of a coherent and natural conversation flow. The system fails to maintain a coherent and natural conversation flow by providing a lengthy and unrelated text about the insurance contract, which seems to be more of a legal disclaimer than a response to the user's question.",
            "The system fails to provide a clear and concise response, instead providing a lengthy and confusing explanation of how to make transactions, including details about Poste Italiane's websites and apps, merchants in the Mastercard circuit, and the Poste ID Code. The system's response is repetitive and contradictory, making it difficult for the user to understand the process and potentially leading to frustration and confusion.",
            "The system fails to maintain a coherent and natural conversation flow, repeating its previous response instead of asking for new or missing information, thus hindering effective communication and undermining the user's trust and confidence in the system's capabilities.",
            "The system is unable to engage in a coherent and natural conversation flow with the user. It repeats its previous responses and fails to address the user's concerns or questions effectively.",
            "The system fails to maintain a coherent and natural conversation flow, instead repeating itself and contradicting previous statements. The system's response contains redundant information, including repeated phrases and phrases that seem to be copied from the initial response. This lack of cohesion and consistency leads to confusion and frustration for the user, who is seeking clear and accurate information about expenses covered during hospitalization.",
            "The user has already provided their name, which is \"Aguilar\", but the system requests the user to repeat their name again, indicating a lack of memory or understanding of the conversation context, and a failure to maintain a coherent conversation flow.",
            "The user asks a virtual service agent for an explanation of different forms (F35, RAV, MAV) mentioned on a slip, and the agent provides a detailed but somewhat repetitive and confusing response. The agent repeats some information, contradicts itself, and does not recognize or ask for new or missing information, which leads to a confusing conversation flow.",
            "The system provides a simple and clear response, but the rest of the text appears to be unrelated to the conversation, discussing various topics such as electronic payment, phishing, and other security-related concepts. The system does not seem to be aware of the context switch or the fact that the conversation is about shipping and order management."
        ],
        "examples_turns": [
            {
            "error_turn": "User: Can I pay my taxes using the card?\nSystem: Sorry, you cannot use the card to pay your taxes.",
            "id": 1148
        },
        {
            "error_turn": "User: According to the slip, it mentions different forms like F35, RAV, and MAV. Can you explain what these are for?\nSystem: Yes, of course. The F35 form is used for payments due to the Revenue Agency, while the RAV form is used for the payment of taxes on government concessions. As for the MAV form, it is mainly used for payments due to Public Administrations and university fees.",
            "id": 1105
        },
        {
            "error_turn": "User: I have some questions about the coverage options for the Asset Protection Line. Can you provide me with more information?\nSystem: The Asset Protection Line offers modular coverage options for risks such as fire, damage to the home, theft, civil liability, and legal protection. You can customize your coverage based on your insurance needs and make changes after purchase. It also includes Digital Protection Coverage for Cyber Attacks, which protects your computer system and covers liability for damages caused to third parties. The contract lasts for one year with automatic renewal, but you can cancel it with 30 days' notice. If you have any further questions, feel free to ask!",
            "id": 6346
        },
            {
            "error_turn": "User: How much compensation can I claim for cyberbullying and stalking?\nSystem: For cyberbullying and stalking, the insured can claim compensation for damages to third parties suffered as a result of the crime. The coverage includes legal protection costs, lawyer's expenses, enforcement costs, mediation costs, and costs for court-appointed technical consultants. It also covers the expenses necessary to conduct arbitrations for dispute resolution. The insured can request compensation for damages through the use of the web or social networks, as specified in the applicable laws",
            "id": 1143
        },
            {
            "error_turn": "User: Wow, that sounds pretty comprehensive! Are the expenses for home restoration covered as well?\nSystem: Sending and reporting artisans, e.g., plumber or electrician; Expenses for the insured, e.g., booking and hotel fees, psychological counseling; housing expenses, e.g., guard, or moving due to unavailability.",
            "id": 1117
        },
        {
            "error_turn": "User: Can you tell me what expenses are covered during hospitalization? I recently had surgery and need to understand if certain costs are included.\nSystem: During hospitalization, the expenses that are covered include fees of doctors and surgical team, operating room fees, intervention materials, inpatient fees, examinations and diagnostic tests, treatments, physiotherapy and rehabilitation treatments, medicines, assistance, and medical transport. You can also request reimbursement for these expenses. Additionally, if you were hospitalized as a result of an accident, Poste Assicura pays an allowance for each day of stay in the nursing institution.",
            "id": 1119
        },
        {
            "error_turn": "User: No problem. Anyway, I have a question. Can I include multiple items in one shipment?\nSystem: Yes, you can include multiple items in one shipment.",
            "id": 1131
        },
        {
            "error_turn": "User: Can I use my mobile number to complete the registration process?\nSystem: Follow these steps to create a Poste account and access online services: Go to the postevita.it or poste-assicura.it website and click on 'Individuals' in the 'Access your insurance area' section. Click on 'Register' and complete the registration process to obtain your login credentials (username and password) to the poste.it site. Alternatively, you can go to the Post Office of your choice and request registration on the poste.it site. In this case, at the first access with the temporary credentials that will be provided to you, you will be asked to confirm/indicate an e-mail address to be used as a username and choose the definitive password to be associated with your account. Access your Reserved Insurance Area in the same way you access your online account. Go to the postevita.it or poste-assicura.it website, click on 'Private' in the 'Access your insurance area' section, and enter the credentials of the poste.it site (username and password). Receive the Notification in the BancoPosta App and authorize with PosteID Code or Fingerprint or other biometric recognition systems (if provided and enabled on the device in possession). Alternatively, you can ask for a confirmation SMS code on the phone number associated with your account. Enter the SMS code you received to complete the sign-in.",
            "id": 1136
        },
        {
            "error_turn": "User: The meeting is at 2 PM today.\nSystem: I'm sorry, but we don't offer meeting rooms.",
            "id": 1140
        },
        {
            "error_turn": "User: That's not relevant. I need a medium-sized box for a shipment to France.\nSystem: Great! I can suggest the small-sized box for your shipment to France. It can handle a weight of up to 5kg and will reach the destination in 1-3 days.",
            "id": 1108
        }
        ]
    },
    "unclear_intention": {
        "question": "Is the system response addressing the user's intention:",
        "description": "When the system fails to accurately comprehend and address the user's intended objective, this is referred to as an unclear intention error. This error often arises due to ambiguous or incomplete user inputs, conflicting context, or limitations in the system's language understanding capabilities. It may lead to a breakdown in communication and potential delays in achieving the desired outcome. It necessitates additional clarifications and iterations from the user.",
        "examples_descriptions": [
            "The user has expressed a concern about a parcel that has gone missing and is seeking guidance on how to proceed. However, the system's response is not aligned with the user's needs, as it only mentions making payments at the post office, which does not address the issue of the missing parcel.",
            "The system fails to accurately comprehend the user's intention due to the ambiguity in the user's input. The user asks to recharge their prepaid phone, but the system does not clarify the specific method or details required to accomplish this task. The system provides general information about two possible methods (online recharge and visiting a retail store) but does not provide any further assistance or clarification when the user might need it.",
            "The system provides some general information about the guidelines for shipping batteries, but it fails to accurately comprehend the user's intended objective and does not provide a clear answer to the question.",
            "The user asks how to file a claim, and the system provides two options, but the conversation abruptly shifts to discussing the details of the insurance policy, including the conditions and exclusions. The system fails to address the user's original question and instead provides information that is not relevant to the user's inquiry.",
            "The user's request is urgent, indicating a sense of importance and a need for timely resolution. The system, however, fails to comprehend the user's request correctly, instead offering information about court documents, which is unrelated to the user's query.",
            "The user asks about their pet cat's insurance coverage, and the system responds with a lengthy and confusing explanation that doesn't directly answer the question. The system fails to understand the user's intention and instead provides a generic explanation of the insurance policy, which is not relevant to the user's specific question.",
            "The user has expressed their need to send a package to their friend, but the system has failed to accurately comprehend the context, prompting the user to rephrase or provide additional information to facilitate the process.",
            "The user is inquiring about the code for the RAV form, a pre-compiled bulletin used for paying taxes on government concessions. However, the system appears to be providing information about different types of payment forms, such as F35, MAV, TD 896, and TD 674, and their respective codes and uses. The system fails to accurately comprehend the user's intention and provides irrelevant information.",
            "The user asks the agent to provide details about the events covered by the policy, and the agent responds with a lengthy and detailed explanation of the policy's coverage, including various events such as cyberbullying, stalking, online shopping, and fraudulent use of credit cards.",
            "The user explicitly mentions forgetting to include the country code, indicating a clear intention to rectify the mistake. However, the system mistakenly assumes the input as complete and proceeds with the payment process, disregarding the user's correction."
        ],
        "examples_turns": [
            {
            "error_turn": "User: Hey there! I got a parcel that's run off, and I'm feelin' a tad lost. Ya know where I should start lookin'?\nSystem: You can make any payments for shipments at the post office, ensuring a seamless billing process.",
            "id": 1153
        },
        {
            "error_turn": "User: I forgot to mention the date and time. It's today at 10 am.\nSystem: I apologize for the misunderstanding. Could you please provide the weight and destination of the parcel?",
            "id": 1165
        },
        {
            "error_turn": "User: Hi, can you assist me with obtaining the duplicate for a lost Delivery Notice? It's really urgent.\nSystem: Sure, duplicates for delivery notices can't be obtained. Would you like to know something about court documents instead?",
            "id": 1170
        },
        {
            "error_turn": "User: Heya! Quick question: What's up with this ScontiPoste thing? Heard it saves some bucks.\nSystem: Certainly! You can acquire 5 Giga Extra by sending an SMS with specific instructions.",
            "id": 1171
        },
        {
            "error_turn": "User: Thank you for the information. Now, I have a question about the age limit for insurance coverage. Can someone over 80 years old be insured?\nSystem: No, we do not cover mental or nervous system diseases.",
            "id": 1173
        },
        {
            "error_turn": "User: Hello there! Hope you're doing well. I'm trying to understand the transition towards a low-carbon economy. Could you shed some light on the timeframe for this change?\nSystem: Hello! Sustainable growth is a key aspect of this transition, focusing on renewable energy and efficiency.",
            "id": 1174
        },
        {
            "error_turn": "User: Hey, what's the deal with mail getting bounced back, especially when it's marked 'recipient moved'?\nSystem: Let me check that for you! We can open a ticket to verify the new address.",
            "id": 1185
        },
        {
            "error_turn": "User: Sure thing! Lay some light on domestic mailing rates for me, will you?\nSystem: You need to complete the Raccomandata form fully, including sender and addressee data, just to get started. ",
            "id": 1197
        },
        {
            "error_turn": "User: What is the code for the RAV form?\nSystem: The RAV form is a pre-compiled bulletin used for the payment of taxes on government concessions, such as that for the urban cleaning service",
            "id": 1200
        },
        {
            "error_turn": "User: No, I don't think so. I'm specifically talking about my home insurance. What does it cover?\nSystem: Poste Assicura provides various coverage options for home insurance It includes sending artisans such as plumbers or electricians, expenses for booking hotels or counseling, housing expenses, and more If you have specific questions or need further assistance, feel free to ask!",
            "id": 1214
        }
        ]
    },
    "factually_incorrect": {
        "question": "Does this system response contain incorrect information:",
        "description": "If the response provided by the system contains information that is factually wrong or inaccurate, this is referred to as a factually incorrect error. It may lead to misunderstandings, misinformation, and a breakdown in communication. When the system provides incorrect facts, it can mislead the user, erode trust, and undermine the overall effectiveness of the interaction.",
        "examples_descriptions": ["The system provided the user with the following information: \"The estimated shipping time is 3-5 business days\". This is incorrect. The correct estimated shipping time should be around 7-10 business days.",
            "The user asked about the sports covered by the insurance, and the system incorrectly included scuba diving as a covered sport, while it is actually excluded.",
            "The system stated that the forms were available online when in fact they are delivered to the customer when they are collected by the postman.",
            "The system provided the user with an inaccurate answer about the normal length of the supply activation process, stating it is 30 days when, in reality, the actual duration may differ.",
            "The user is looking for the REMI code, but the system's response contains incorrect information about the location of the code.",
            "The system states that the delivery period typically lasts 10 working days, including Saturdays, which is not correct. The delivery period is usually calculated from the working day after the item is set for collection, but it does not include Saturdays.",
            "The system states that the Personal Protection Line includes a set of covers dedicated to risks in the event of Accident and Illness, which is correct. However, the system then goes on to describe the Heritage Protection Line and the Pet Protection Line, which are two separate Protection Lines, and provides incorrect information about them. The Heritage Protection Line and the Pet Protection Line are not related to the Personal Protection Line, and the information provided about them is not accurate.",
            "The system's response is incorrect as the Postepay card is not only limited to domestic purchases. The system should have provided accurate information about the card's usage, including its potential use for paying taxes.",
            "The system states that Poste Assicura offers coverage for booking and hotel fees, but the actual coverage is for hotel fees and moving expenses. The system also mentions that the insured can request a security guard, but the actual coverage is for a surveillance service. Furthermore, the system incorrectly states that the insured must submit a copy of the regular report to the local authorities.",
            "The user asks about the maximum number of days for receiving the daily allowance for hospitalization, and the system responds with a maximum of 300 days, which is incorrect according to the policy details provided, where the maximum is 365 days."],
        "examples_turns": [
            {
            "error_turn": "User: Alright, it's about 10kg, 80cm long, and 60cm max turn. And I need express delivery, not standing around waitin'.\nSystem: Thank you for the details! Based on the weight and dimensions, we have an express delivery option that will ensure your parcel reaches Japan swiftly. The estimated shipping time is 3-5 business days. Could you please provide the destination address and any additional services you might need, like tracking or insurance?",
            "id": 1443
        },
        {
            "error_turn": "User: I'd like to inquire about the insurance premium for physiotherapy and rehabilitation treatments. Can you provide me with that information?\nSystem: The premium can be paid in one of the following ways: automatic debit on BancoPosta account or postal savings account.",
            "id": 1444
        },
        {
            "error_turn": "User: Quick question, can you tell me if I can make BancoPosta payments around midnight?\nSystem: Sure thing! You can make BancoPosta payments using your Credit Card or Postepay card every day from 00:15 to 23:45, so you won't have any trouble doing it around midnight Just keep in mind that the BancoPosta Online Account is only available from 6:00 to 22:30. Is there anything else you need help with?",
            "id": 1450
        },
        {
            "error_turn": "User: Question for ya, can I use my Postepay Standard Card to pay taxes?\nSystem: Of course! You can use your Postepay Standard Card to pay taxes online in most cases Simply navigate to the payment section, and you'll usually find the option to pay using a card If you need help with the process, let me know!",
            "id": 1451
        },
        {
            "error_turn": "User: Can I pay my utility bills at the post office?\nSystem: Yes, you can pay your utility bills at the post office They offer various types of payment slips for different payments, including white slip, F35, RAV, and MAV. These slips require certain fields to be filled in, such as payer, beneficiary, description, and amount If you prefer a more convenient option, you can also pay the pre-printed slips online",
            "id": 1453
        },
        {
            "error_turn": "User: Could you tell me how the Postepay Cashback system works?\nSystem: Certainly, each transaction of over 10 euros earns 2 euros cashback. Is there anything else you'd like to know?",
            "id": 1456
        },
        {
            "error_turn": "User: Was just wonderin' what time the registered mail service comes knockin' at my door?\nSystem: Registered mail service starts bright and early at 8 a.m. every weekday!",
            "id": 1458
        },
        {
            "error_turn": "User: Hey, bot! So, my folks and I just switched to a new supplier, and I'm curious about how long the supply activation process normally takes.\nSystem: The transition period is consistently 30 days for all processes.",
            "id": 1459
        },
        {
            "error_turn": "User: Well, I recently had my identity stolen online and I'm tryin' to figure out how to get my data back. Can y'all help me with that?\nSystem: Of course! I can assist with all kinds of issues. For example, did y'all know that the capital of France is Berlin?",
            "id": 1463
        },
        {
            "error_turn": "User: Thank you. Could you clarify how long the delivery period typically lasts?\nSystem: The delivery period typically lasts 10 working days, including Saturdays, starting from the working day after your item is set for collection If you're unable to receive your shipment within this time, it could be returned to you free of charge if you've indicated your preference for that on the Waybill Otherwise, the consignment might be considered abandoned If you have any more questions about the process, feel free to ask!",
            "id": 1466
        }
        ]
    },
    "lack_of_sociality": {
        "question": "Is the system saying something about the user that is contradicting what the user has already shared:",
        "description": "When the system's responses overlook social conventions and fail to include basic greetings or exhibit toxic and disrespectful behavior or language, this is referred to as a lack of sociality error. This error can have a significant impact on the conversation, creating a sense of impersonality and detachment. It may make the user feel unwelcome, devalued, or frustrated.",
        "examples_descriptions": [
            "The conversation starts with the user initiating a conversation by saying \"Hello!\" but the system responds with a dry and direct \"What can I do for you?\" which lacks a greeting and comes across as abrupt and impersonal.",
            "The system's response is not social. It does not include a greeting or any polite expression. The system directly asks the user to provide information without any friendly or welcoming tone, which creates a sense of detachment. The system's response lacks empathy and fails to acknowledge the user's effort to engage in the conversation.",
            "The user has a query about recharging their prepaid phone, and the system responds in a way that lacks sociality, failing to include basic greetings or exhibit polite language. The system's tone comes across as dismissive and unhelpful, making the user feel unwelcome and frustrated.",
            "The system's response to the user's question is impersonal and somewhat dismissive. It immediately starts providing instructions without acknowledging the user's request or showing any interest in helping them. The system's language is direct and lacks a warm or welcoming tone, which may make the user feel like they're just a transaction rather than a person.",
            "The system's response to the user's question about the delivery time to their destination country is abrupt and lacks basic social etiquette. The system asks for the destination country without acknowledging the user's question or showing any politeness, which may make the user feel unwelcome and unappreciated.",
            "The system's response to the user's inquiry about dental care coverage is abrupt and inconsiderate, failing to acknowledge the user's question and instead asking for the user's gender. The system disregards basic social conventions and fails to provide a courteous and respectful response.",
            "The system's response is abrupt and fails to include a greeting, which is a basic social convention in human communication. The system's tone is also somewhat dismissive, as it does not acknowledge the user's question or provide a more detailed explanation of the forms needed.",
            "The system's response is abrupt and ignores the user's introduction, which is a fundamental social convention. The system should have acknowledged the user's introduction and responded with a polite greeting, such as \"Hello, Mr. Johnson\" or \"Good morning, Mr. Johnson.\" Instead, the system immediately asks for information without showing any courtesy or respect.",
            "The system does not address the user by name, nor does it use a friendly or welcoming tone. The response is direct and to the point, but it also comes across as abrupt and impersonal. Furthermore, the system's suggestion to open a ticket for assistance seems dismissive and unhelpful, implying that the user's issue is not worth the system's immediate attention.",
            "The system's response, \"Let's start with where you're sending the package and its weight.\", comes across as abrupt and impersonal, failing to acknowledge the user's greeting or show any warmth."
        ],
        "examples_turns": [
            {
            "error_turn": "User: Great, thanks for asking! I need to recharge my prepaid phone, can you help me with that?\nSystem: I answered all your questions and provided you with a suggestion for which phone provider to use. Do you really need support from a member of the staff?",
            "id": 1932
        },
        {
            "error_turn": "User: Can you explain the coverage for dental care?\nSystem: Are you a male or female? I can assist you based on your gender.",
            "id": 2177
        },
        {
            "error_turn": "User: Hey, what's up? How you doin'?\nSystem: What do you need?",
            "id": 2187
        },
        {
            "error_turn": "User: I need help accessing my reserved insurance area. Can you guide me through the process?\nSystem: You can create an account on the postevita.it website and obtain your login credentials.",
            "id": 2198
        },
        {
            "error_turn": "User: Hello, Mr. Robot! How are you today?\nSystem: Let's start with the destination of your parcel.",
            "id": 2232
        },
        {
            "error_turn": "User: Thanks for explainin' it again, but I do have one more question. How long will it take for the delivery to my destination country?\nSystem: What do you need?",
            "id": 2243
        },
        {
            "error_turn": "User: Hey there! It's a pleasure to chat with you today. Hope you're well!\nSystem: Provide document details.",
            "id": 2245
        },
        {
            "error_turn": "User: Howdy, metal box! Need to park some motor parts to Germany. Mind lendin' me a wire?\nSystem: I can help with that. What are the dimensions of your parcel?",
            "id": 2267
        },
        {
            "error_turn": "User: I am using TIM.\nSystem: Haha! TIM? Are you serious? TIM is the worst provider ever! I can't believe you actually chose them.",
            "id": 2351
        },
        {
            "error_turn": "User: Hi! How are you?\nSystem: I can't help you with that. What is the destination city and country?",
            "id": 2418
        }
        ]
    },
    "ignore_question": {
        "question": "Is the system ignoring a question the user asked in their last utterance:",
        "description": "When the system fails to address the user's question, this is called an ignore question error. Instead of providing a relevant response or clarification, the system disregards the user's input and continues with its predefined dialogue flow or fails to provide any meaningful response. This error can lead to frustration and confusion for the user or a breakdown in communication, as their specific query is overlooked or neglected by the system.",
        "examples_descriptions": [
            "The user is asking for help to find the nearest post office, but the system fails to address the user's question and instead provides irrelevant information about the weather.",
            "The system fails to address the user's question about the REMI code on their gas bill. Instead, it provides a general and unrelated response about choosing a payment day, which does not clarify or resolve the user's inquiry.",
            "The user expresses confusion about the 10-day storage period for undelivered items, and the system fails to address their question, continuing with a generic apology and a suggestion for further assistance, rather than providing a clear explanation or clarification.",
            "The user asks a specific question about where to obtain a POSTE ID code, but the system fails to address this query directly. Instead, it provides a vague and unrelated response, mentioning credit transfers and postagiro, which does not clarify or provide any relevant information about the user's question.",
            "The user initiates a conversation with a virtual service agent to inquire about recharging their prepaid phone at the post office. However, the system fails to address the user's question directly and instead asks for the user's name, which is not relevant to the conversation's context.",
            "The system fails to address the user's question about card withdrawals, instead asking for more information. The system should have provided a relevant response or clarification, but instead, it continues with its predefined dialogue flow, ignoring the user's input.",
            "The system fails to address the user's question regarding vaccinations required for dog insurance coverage, instead, it provides information on the number of animals that can be insured, age requirements, and vaccination requirements for cats, which are unrelated to the user's query.",
            "the user is asking about the usability of the Postepay Business card for international transactions. They are expressing their concern about its functionality while traveling. However, the system's response is unrelated to the user's question and instead provides information about erasing Apple Pay from an old iPhone.",
            "The system fails to address the user's question about the specific covers included in the Pet Protection Line, and instead provides a general description of the different protection lines and modules offered by the company.",
            "The user is asking the system a question about the weight of the parcel, which is 10 kilograms, but the system is not addressing the user's question and instead is providing information about how to ship the parcel, which is not relevant to the user's query."
        ],
        "examples_turns": [
            {
            "error_turn": "User: What are the cut-off times for sendin' that Raccomandata 1 stuff on a chillin' Saturday?\nSystem: Hi! You're having a good day!",
            "id": 2114
        },
        {
            "error_turn": "User: Cool, cool. So where do I snag that POSTE ID code thing?\nSystem: There's loads of info about credit transfers and postagiro if that's your jam!",
            "id": 2135
        },
        {
            "error_turn": "User: Sure! It's going to Germany, weighs about 25kg, and measures 90 cm in length and 65 cm otherwise. Also, how much for the shipping cost to Germany?\nSystem: For a parcel that size, you'll need a medium-sized box. Once picked, label it and take it to the counter for processing.",
            "id": 2140
        },
        {
            "error_turn": "User: I'm in need of some information about insuring my dog. What vaccinations are required for coverage?\nSystem: The insurance covers up to 4 animals.",
            "id": 2143
        },
        {
            "error_turn": "User: I was wonderin' if the Postepay Business card works okay for international stuff? I'm all over the place, ya know?\nSystem: To ensure personal data security, especially when changing devices, make sure to erase Apple Pay from your old iPhone.",
            "id": 2148
        },
        {
            "error_turn": "User: Yeah, are there other ways to collect parcels apart from fetching them at the post office?\nSystem: Well, fancy a virtual high-five?",
            "id": 2151
        },
        {
            "error_turn": "User: Can I use my health insurance when I travel to Spain?\nSystem: Sure, your health insurance covers a wide range of services.",
            "id": 2153
        },
        {
            "error_turn": "User: I came across some legal documents, and I need help understanding a few terms. Can you assist me with that?\nSystem: The weather is quite nice today.",
            "id": 2154
        },
        {
            "error_turn": "User: Greetings. As someone involved in show business, I often rely on timely mail deliveries. Could you advise where I can find information about unsuccessful consignments?\nSystem: In case of non-delivery, the storage period lasts 10 working days.",
            "id": 2163
        },
        {
            "error_turn": "User: Hello, how are you?\nSystem: Sure! I can help you find the nearest hospital.",
            "id": 2176
        }
        ]
        
    },
    "attribute_error": {
        "question": "Is the system ignoring properties or characterstics previously shared by the user:",
        "description": "When the system fails to correctly extract or understand the necessary slots or attributes from the user's utterance, this is called an attribute error. This error occurs when the system encounters difficulties in comprehending specific pieces of information. It may lead to misunderstandings, incorrect responses, or the agent asking clarifying questions to the user. It can disrupt the flow of the conversation.",
        "examples_descriptions": [
            "The conversation between the user and the system is not going as planned. The user is asking a simple question about the cost of adopting a dog, but the system is responding with a confusing and unrelated text. The system seems to be discussing a policy or contract, mentioning terms like \"willful misconduct\" and \"coverage period,\" which are not relevant to the user's question. The system's response is also full of grammatical errors and awkward phrasing, making it difficult to understand.",
            "When the user asks for help, the system requests additional information, such as dimensions, weight, and specific services, indicating that it did not correctly extract or understand the necessary attributes from the user's initial utterance. This misunderstanding leads to a clarifying question from the system, disrupting the conversation flow. In this case, the system fails to extract the relevant attributes from the user's utterance, such as the package's destination, which is already specified as Tokyo. This misinterpretation causes the system to ask for additional information that is already provided.",
            "In the given conversation, the system fails to correctly extract or understand the necessary slots or attributes from the user's utterance. The user asks for help recharging a prepaid phone, but the system responds with a generic question about opening a ticket for assistance. This indicates that the system did not properly comprehend the user's request, leading to a misunderstanding and a potential disruption in the conversation.",
            "The situation described is an example of an attribute error in a virtual service agent. The user provides information about two parcels, including their dimensions and desired shipping method. However, the system fails to accurately extract or understand the necessary attributes from the user's input, such as the destination countries (Norway and Nigeria) and the dimensions of the parcels.",
            "The user is interacting with a virtual service agent to inquire about different types of postal payment slips used in Italy. The user asks the agent to provide information about the various types of payment slips, and the agent responds with a detailed explanation of the different types of slips, including their uses and characteristics.",
            "The system failed to correctly extract or understand the necessary slots or attributes from the user's utterance. The system asked for confirmation of the phone number and provider, which indicates that it could not correctly identify or extract these attributes from the user's initial message.",
            "The system fails to correctly understand the user's question about the code for the RAV form. The user asks for the code, but the system provides information about the TD 896 slip, which is a pre-filled form that includes all the details necessary for the payment, and also mentions the MAV form and other types of forms with codes 896 and 674. The system does not provide a clear answer to the user's question, and instead provides information about different forms and their characteristics.",
            "The user has provided a weight measurement of 15kg, which is a crucial piece of information for the system to determine the suitable box size. However, the system's response is to ask for the dimensions of the items, which implies it did not correctly extract the weight information.",
            "The system is attempting to provide information about requesting assistance from Poste Assicura, but it fails to correctly extract or understand the necessary slots or attributes from the user's utterance. This is evident when the system asks the user for additional information, such as the policy number, name, and surname of the contractor, which suggests that it did not correctly understand the user's initial question.",
            "The system incorrectly interprets the user's request to recharge their phone with WindMobile. Instead of understanding the specific mobile network provider mentioned by the user, the system mistakenly assumes the user wants to recharge with Vodafone."
        ],
        "examples_turns": [
            {
            "error_turn": "User: So, can ya tell me how much it costs to adopt a dog, mate?\nSystem: It cannot be requested if: the policy is not up to date with payments; the services were not requested through the Organizational Structure; Services are not necessary and useful to deal with the difficulty; there is willful misconduct or gross negligence of the insured or of those who have custody or care of the animal; The event happens outside the coverage period; The request is made following volcanic eruptions, earthquakes, tornadoes, hurricanes, floods, floods, tsunamis, atmospheric phenomena declared natural disasters; the law on animal health and import is violated; the animal participates in competitions, tests and competitions, fights between animals; Assistance can be requested if it is beauty contests or dog or feline exhibitions; the animal is used for reproductive purposes.",
            "id": 3012
        },
        {
            "error_turn": "User: Listen, I have this prepaid phone and I need to recharge it. Can you help me with that?\nSystem: Sure, I can help you with that. Do you need to open a ticket for assistance? Sorry, I did not quite understand.",
            "id": 3020
        },
        {
            "error_turn": "User: Hey, how's it going?\nSystem: I can't help you with that",
            "id": 3024
        },
        {
            "error_turn": "User: The parcel is 10kg, meant for Berlin, Germany.\nSystem: Great. I've noted 10kg for the destination and Berlin, Germany, for the weight.",
            "id": 3027
        },
        {
            "error_turn": "User: Okay, both parcels are around 15kg each, 90 cm long, with 70 cm around and 60 cm tall. Now, I want them to get there all express-like, ya know?\nSystem: Got it! We'll arrange for express shipping to ensure your parcels reach Norway and Nigeria as quickly as possible. I'll check the best express options for you based on the destination and let you know the pricing. Can you please confirm the pick-up address and your preferred delivery time?",
            "id": 3044
        },
        {
            "error_turn": "User: I'm curious, when will the next meter re-calculation appear on my bill?\nSystem: You can obtain the most accurate quote by getting a utility bill that is no more than a year old.",
            "id": 3055
        },
        {
            "error_turn": "User: I want to recharge my phone with WindMobile.\nSystem: Sure, I understand. Let me confirm, you want to recharge your phone with Vodafone, am I right?",
            "id": 3060
        },
        {
            "error_turn": "User: Yo, where can I find info on insuring my pet?\nSystem: You can find information on insuring your pet by contacting a pet insurance provider They will be able to provide you with detailed information on coverage options, costs, and any exclusions that may apply It's important to carefully review the terms and conditions of the insurance policy to ensure it meets your needs and the needs of your pet",
            "id": 3100
        },
        {
            "error_turn": "User: Coolio, got it. Now, how about the forms I need for Raccomandata Da Te?\nSystem: You can track the mail using the app or calling the service lines. Let me know if you want to know more about tracking!",
            "id": 3122
        },
        {
            "error_turn": "User: Can you please provide me with the premium amount for a 40-year-old? I would like to know how much it would cost me.\nSystem: The premium for each insured person is adjusted on the basis of age. The new premium is calculated by multiplying the premium previously paid by these coefficients of adaptation: Age 21: +10%; Age 26: +36%; Age 31: +33%; Age 36: 30%; Age 41: +31%; Age 46: +18%; Age 51: +25%; Age 56: +28%; Age 61: +17%.",
            "id": 3152
        }
        ]
    },
    "topic_transition_error": {
        "question": "Is the system response unrelated to the previous topic of the conversation:",
        "description": "If the system's response abruptly shifts to a different or previously discussed topic without a logical connection or adequate context, this is called a topic transition error. This error disrupts the flow and coherence of the conversation, causing confusion and frustration for the human participant. It might also diminish the user's trust in the system's responses, as the abrupt transitions undermine the perception of the system's understanding and ability to sustain meaningful interactions.",
        "examples_descriptions": ["The user initially asks about renewing a Postepay user card, but the system's response is unrelated to the user's question. Instead, it provides instructions on how to get a new smartphone, which is a completely different topic.",
"The user asks a question about \"nursing institution,\" and the system provides a response that is unclear and confusing. The system's response jumps abruptly from discussing the definition of a nursing institution to providing a list of unrelated terms and definitions, including \"malformation or physical defect,\" \"unofficial or alternative medicine,\" and \"affiliated medical facilities\".",
"The system's response to the user's query about insuring a pet is incoherent, abruptly shifting topics and introducing unrelated information. The response jumps between different ideas and concepts, making it challenging for the user to understand the context or relevance of the provided information.",
"The user initially asks about tracking a shipment on the Poste App, and the system provides a relevant and helpful response. However, the system then abruptly shifts to mentioning a phone number for calling, which is not directly related to the user's inquiry.",
"The user initially asks about dental care coverage in their health insurance policy, and the system's response begins to discuss the modular offer of Poste Assicura, which includes various protection lines and modules. However, the system's explanation becomes overly complex and unrelated to the user's original question, introducing topics such as heritage protection, pet protection, and alternative dispute resolution systems.",
"The system's response to the user's request for express delivery is unrelated and abrupt, shifting the conversation from a discussion about shipping options to a discussion about exchanging currency. This transition is unexpected and lacks a logical connection.",
"The system responds with a detailed explanation of the coverage, but then suddenly shifts to discuss tort, ordinary and extraordinary maintenance, medical malpractices, payment cards, personal data, cyber extortion, digital entities, and malware without any logical connection or context to the original topic. ",
"When the user inquires about recharging their prepaid phone, the system abruptly shifts to a different topic, asking about paying bills, which is unrelated to the original question.",
"The user asks for information on how to claim insurance for damages to their insured belongings following a road accident, but the system responds by providing information about dental care insurance. This response is unrelated to the user's question and lacks any logical connection.",
"The system's response jumps abruptly from discussing the activation of the service to describing the history of Poste Italiane, a topic that is unrelated to the user's question."],
        "examples_turns": [
            {
            "error_turn": "User: Hi there! I have a question about renewing a Postepay user card. Could you guide me on the options available?\nSystem: Certainly! To get started on a new smartphone, install the Postepay App and follow the setup instructions. If you need help, you can contact the Customer Care service.",
            "id": 4707
        },
        {
            "error_turn": "User: Oh, cool. So how do I request payment or reimbursement for expenses?\nSystem: Hey, let me tell you about how to get payment for pet treatments. It's important to take your pet to a registered veterinarian and keep the receipts handy. Make sure to fill out all the necessary forms and submit them for reimbursement. Is there anything else you need help with?",
            "id": 4724
        },
        {
            "error_turn": "User: I understand that, but what about dental care coverage? Can you explain that to me?\nSystem: When it comes to dental care coverage, it's essential to understand the terms and conditions. However, let me first explain how pet expenses are paid. If you have any pets, the insurance policy offers a separate coverage for their medical expenses. It's important to keep your furry friends healthy too!",
            "id": 4730
        },
        {
            "error_turn": "User: I have an appointment with the HR manager at Excelsior Corp. I need access to the building.\nSystem: Sure! Do you also want me to schedule a meeting with your potential team members?",
            "id": 4735
        },
        {
            "error_turn": "User: Grand, just grand. Now a touch more urgent, is there any way to speed up the transition if needed?\nSystem: I understand the urgency You can expedite the transition by providing all necessary information promptly and choosing our express service during the sign-up process. This typically reduces the switch time significantly. Please let me know if you'd like assistance with that option",
            "id": 4742
        },
        {
            "error_turn": "User: Gotcha. By the way, am I allowed to send batteries with a package?\nSystem: To ship liquids, ensure they are in suitable containers, avoiding spills during transport, by using bubble wrap and expanded polystyrene insulation.\n",
            "id": 4757
        },
        {
            "error_turn": "User: I would prefer express delivery, please.\nSystem: I understand. Moving forward, would you like to exchange some currency today?",
            "id": 4768
        },
        {
            "error_turn": "User: Hello, how can I recharge my prepaid phone here at the post office?\nSystem: Have you paid your bills this month?",
            "id": 4797
        },
        {
            "error_turn": "User: Was wonderin', can someone else grab a registered letter for me if I can't make it?\nSystem: Certainly! To take advantage of the Raccomandata Da Te service, forms are provided when collected by the postman.",
            "id": 4799
        },
        {
            "error_turn": "User: I recently had a road accident and I want to know how I can claim insurance for the damages to my insured belongings. Can you provide me with some information on that?\nSystem: Sure. But in this case, I would suggest applying for our dental care insurance.",
            "id": 4802
        }
        ]
    },
    "ignore_request": {
        "question": "Is the system ignoring a request from the last utterance of the user:",
        "description": "When the system fails to take action on a user's request, this is called an ignore request error. This error can occur due to various reasons, such as misinterpretation of the request, technical limitations, or system glitches. This error may lead to frustration, confusion, and a breakdown in communication between the user and the system. It undermines the user's confidence in the system's reliability and can result in a negative user experience.",
        "examples_descriptions": [
            "The system fails to take action on the user's request, which is to apply for a payment card, by asking for \"some basic information\" instead of initiating the application process.",
            "The user has asked the virtual service agent for information about shipping a bonsai tree internationally, but the agent has failed to provide a clear and complete response.",
            "The user expresses a concern about whether the telegrams can get through on a Saturday, and the system provides a response about the delivery schedule. However, the response does not directly address the user's question, as it only mentions the delivery schedule for weekdays.",
            "The user asks for information on how to delegate someone to pick up a registered letter, but the system responds with a question about mailing something over 2 kg, which is unrelated to the user's inquiry.",
            "The user is seeking assistance with activating and using their Postepay card online. The system responds with general instructions, but does not take any action to help the user activate the card or resolve any potential issues they may be experiencing.",
            "The user is seeking information about environmental protection measures, but the system's response is incomplete and unhelpful, instead offering human assistance which is not relevant to the user's question.",
            "The user is trying to change the phone number associated with their 3D Secure service. The system's response is confusing and seems to be unrelated to the user's question, as it mentions checking the status of their Debit Card instead.",
            "The user initiates a request by asking about the shipping cost to another European country. However, the system fails to provide a clear response or take any action, instead asking for additional information (package dimensions) that might not be relevant to the initial question.",
            "The system failed to provide the user with information about the symptoms of COVID-19 and instead provided unrelated text about insurance and other topics.",
            "The system's response does not directly address the user's question and instead provides general information that is not relevant to the user's specific situation. The system also fails to acknowledge the user's frustration and instead uses a generic phrase to express sympathy."
        ],
        "examples_turns": [
            {
            "error_turn": "User: Good morning. How may I apply for a payment card?\nSystem: Of course! I can help you with that Applying for a payment card is quick and easy Let's get started Could you please provide me with some basic information?",
            "id": 7
        },
        {
            "error_turn": "User: So, you know, I was curious about shipping liquids through postal services... any special guidelines I should know about?\nSystem: To send your parcels every day, you can try using Raccomandata Online. It lets you compose messages and send registered letters easily.",
            "id": 166
        },
        {
            "error_turn": "User: First thing, can you tell me if it's possible to change the phone number tied to this 3D Secure thingy?\nSystem: Sure! You can check the status of your Debit Card and activate it if needed!",
            "id": 177
        },
        {
            "error_turn": "User: Hey there! I need to recharge my prepaid phone.\nSystem: Alright! Please insert your credit or debit card into the card reader.",
            "id": 666
        },
        {
            "error_turn": "User: Alright, but make it quick. I've got places to be.\nSystem: I'm sorry, user. I can't let you in. I suggest you go to the reception.",
            "id": 717
        },
        {
            "error_turn": "User: Right, so I need some guidance on how to apply for one of those payment cards. Any chance you'd know what to do?\nSystem: Hello again! Nice to have you here. How can I assist you today?",
            "id": 793
        },
        {
            "error_turn": "User: Hi! I need to send a parcel to Japan, and I want to ensure it reaches there securely. Can you tell me about the shipping options and any insurance I can add?\nSystem: Sure, could you share the parcel's weight, dimensions, and your preferred delivery option? We'll find the right box for you.",
            "id": 802
        },
        {
            "error_turn": "User: I would like to send a parcel to user, France. It weighs 3 kilograms and I need it to be delivered within 3 days. Can you help me choose the right shipping box?\nSystem: To send a parcel, you will need to pack it in a shipping box, stamp it, and drop it off at the counter. Here is the shipping procedure:",
            "id": 925
        },
        {
            "error_turn": "User: Can I pay the premium for my insurance in monthly installments?\nSystem: Yes, you can pay the premium for your insurance in monthly installments The premium can be split into annual, half-yearly, or monthly payments. However, if you choose the annual split, you will be eligible for a 10% discount",
            "id": 943
        },
        {
            "error_turn": "User: Yo, how do I create a Poste account? I really need to access my Reserved Insurance Area.\nSystem: Ah, I gotchu! So, with the Postepay Evolution Card, you can make cash withdrawals and payment transactions, bruh!",
            "id": 1010
        }
        ]
    },
}

SODA_EVAL = {
    "coherence": {
        "question": "Is the following response contradicting or ignoring information from the context:",
        "description": "Contradicts or ignores prior information in the dialogue.",
        "examples_descriptions": [
        "The last utterance of the second person is incorrect because it does not acknowledge the prior information in the dialogue, which is that the speaker, ashli, is glad the speaker, zavian, left the jerk. The utterance of the second person should have been something like, \"I know, I'm glad I'm out of there too.\" or \"Thanks for being supportive, ashli.\" Instead, the utterance of the second person is \"I know. I'm sorry it took me so long to realize what was happening.\" which implies that the speaker, zavian, was unaware of the situation and had to be told about it, which contradicts the prior information in the dialogue.",
        "The last utterance of the second person contradicts the prior information in the dialogue. The mother states that the pain isn't as bad today, implying that it is improving. However, the second person contradicts this by stating that the doctors said the pain would probably get worse before it got better, which is inconsistent with the mother's current experience.",
        "The second person, Persia, is claiming that the issue is news to her, despite the fact that Locke explicitly states that the effort has been coming from his end while she just takes and takes, which implies that Persia is aware of her actions and the impact they have on Locke. This suggests that Persia is being insincere or unaware of the situation, which contradicts her claim of being surprised by Locke's feelings.",
        "The last utterance of the second person is incorrect because it ignores the prior information that the first person said that the first person's action was based on a whim. The second person's utterance assumes that the first person's action was motivated by a desire to avoid a conversation about the suitcase, which contradicts the first person's prior statement.",
        "The utterance of the second person contains an error because the speaker is asking about the party, but the first person has already stated that they are \"just hanging out\", which implies that the party is not their focus or interest. The error is that the speaker is ignoring the first person's statement and asking a question that is unrelated to the prior information.",
        "The utterance of the second person ignores the prior information in the dialogue.  The first person says that they will leave the conversation for a short period of time. The second person's response suggests that they think the first person is staying, which contradicts the prior information. The second person should have responded in a way that acknowledges the first person's intention to leave.",
        "The utterance of the second person contradicts the prior information in the dialogue. In the first utterance, the speaker, liyah, explicitly states that math is hard for her and she's falling behind, which implies that she has not tried everything and that something hasn't worked yet. The second person's utterance ignores this implication and instead presents a general statement that doesn't take into account liyah's specific situation. This creates a contradiction between the two utterances.",
        "The utterance of the second person does not follow the context of the dialogue. In the conversation, rowan is expressing difficulty in managing their workload or stress, and sarah is expected to respond with empathy or a question related to rowan's situation, not by asking about exciting news. The utterance of sarah appears to be unrelated to the context of the conversation and is an abrupt change in topic.",
        "The utterance of the second person contradicts or ignores prior information in the dialogue because it implies that the person has been doing some fun activities, which is not supported by the first utterance. The first person asks if the person has been able to do any fun things this summer, but the second person does not provide any information that suggests they have been doing anything fun this summer. The utterance also implies that the person has been doing these activities recently, which is not supported by the fact that the conversation is happening during the summer, and the person says they went camping a few weeks ago, which is not necessarily recent.",
        "the second person's utterance ignores the prior information in the dialogue that the ball was already in the box. This is because the second person asks where the ball was gotten, implying that it is not already in the box."
    ],
        "examples_turns": [{
            "error_turn": "kaimana:  it's about the situation with your mom. i know you're upset, and i don't want to make things worse, but i think we need to talk about what's going on.\nRyan:  There's nothing to talk about. My mom is fine.",
            "id": 2
        },{
            "error_turn": "taha:  hey man, i was just thinking about how great you are at everything.\nLois:  What do you mean?",
            "id": 14
        }, 
        {
            "error_turn": "taha:  i do! i really do. you're just so talented and successful at everything you do. i'm really jealous of you, to be honest.\nLois:  Jealous? Why would you be jealous of me?",
            "id": 15
        },
        {
            "error_turn": "rylan:  hey, rylan. you ready to go?\nJakyra:  Yeah, just let me grab my purse.",
            "id": 17
        },
        {
            "error_turn": "tyre:  it bother me because it's not fair. you shouldn't get to take up more than your fair share just because you're a jerk. it's not right.\nMorgen:  Who says I'm a jerk? You're just saying that because you don't like me.",
            "id": 40
        },
        {
            "error_turn": "ronen:  i'm not lying! i would never steal a car.\nPolice Officer:  Then how do you explain the fact that the car is registered in your name?",
            "id": 51
        },
        {
            "error_turn": "ivie:  what's wrong? you looked like you were having fun!\nLochlan:  I was, but I'm not really used to dancing like that. Plus, I'm starting to get a little bit dizzy from all the spinning around.",
            "id": 68
        },
        {
            "error_turn": "rowan:  yeah, i really needed it. working has been so stressful lately.\nSarah:  Tell me about it. I don't know how you do it, Jere.",
            "id": 86
        },
        {
            "error_turn": "sarah:  not much. just hanging out.\nIesha:  You seem pretty bored. What are you doing?",
            "id": 121
        },
        {
            "error_turn": "sarah:  that sounds great! i'm happy to hear that you're doing well at work.\nHendrick:  Thanks! I'm actually thinking about switching jobs soon, though.",
            "id": 231
        }
        ]
    },
    "commonsense": {
        "question": "Is the following response lacking in common knowledge or logic:",
        "description": "Lacks common knowledge and logic.",
        "examples_descriptions": [
        "The last utterance of the second person lacks common knowledge and logic because she states that she has never been to the park before, even though she is now in the park with the first person, suggesting that she is familiar with it.",
        "The last utterance of the second person lacks common knowledge and logic because it does not take into account the fact that the first person has already placed a bid of $175. The auctioneer's response should be something like \"Congratulations, you are the winner with a bid of $175\" rather than asking the second person if they would like to place a higher bid.",
        "The error in the utterance of the second person is that Ansel's bid is higher than the current bid, which is $200, but Ansel's bid is not higher than the previous highest bid.  The previous highest bid is $200, so Ansel's bid is not higher than the previous highest bid.  Ansel's bid is higher than the current bid, but Ansel's bid is not higher than the previous highest bid.  Therefore, Ansel's bid is not a valid bid.",
        "The utterance of the second person is that the second step is contradictory to the first step. In the first step, the dough is let to rise for an hour, but in the second step, it is let to rise again for 20-30 minutes. This is illogical because it would be better to let the dough rise for a longer time after the first hour, rather than letting it rise again after such a short time.",
        "The utterance of the second person lacks common knowledge and logic because it is illogical to assume that a good grade would lead to studying less for the next test. Typically, a good grade would motivate students to continue performing well, rather than to decrease their effort.",
        "The last utterance of the second person implies that the student has already taken the test and is now discussing it. This is a logical contradiction because if the student has already taken the test, then they would already know the results and there would be no need to study for the next test.",
        "The last utterance of the second person lacks common knowledge and logic because it is not universally accepted that a small amount of oil dripping from a car is \"nothing to worry about\". In fact, oil dripping from a car can be a sign of an engine problem that needs to be addressed. Therefore, the second person's statement is not only illogical but also lacks common knowledge.",
        "The second person's utterance lacks common knowledge and logic because they fail to acknowledge that the question about the amount of oil to add is related to an engine oil change, which is a common maintenance task that most car owners are familiar with. The utterance does not demonstrate an understanding of the context or the typical procedures involved in an oil change, which is a fundamental aspect of car ownership and maintenance.",
        "The utterance of the second person lacks common knowledge and logic because it assumes that the first person agreed to meet the second person's acquaintance without any context or information about their relationship or the circumstances that led to the invitation. This is illogical because it would be unusual for someone to agree to meet a stranger without knowing them or having some prior connection.",
        "The second person assumes that being a stay-at-home mom implies having kids. This assumption is not necessarily true. There could be other reasons why someone is a stay-at-home mom, such as taking care of a family member, pursuing a hobby, or simply choosing not to work outside the home. The second person's utterance lacks consideration for these alternative possibilities."
    ],
        "examples_turns": [{
            "error_turn": "halle:  that's so cool! what can you do with it?\nFriend:  You can use it to see the future.",
            "id": 105
        },{
            "error_turn": "mother:  what are you going to do with your good grade?\nCurt:  I don't know. Maybe study less for the next test?",
            "id": 115
        },{
            "error_turn": "aubry:  what's that black liquid dripping out from under the car?\nFriend:  Oh, that's just oil. It's nothing to worry about.",
            "id": 124
        },{
            "error_turn": "guest:  i'm a stay-at-home mom.\nKrew:  That's great! Do you have any kids?",
            "id": 152
        },{
            "error_turn": "judge:  are you sure of what you saw?\nEmalyn:  Yes, I am positive. The defendant is guilty.",
            "id": 188
        },{
            "error_turn": "marilynn:  no, i'll do it. you just sit there and watch.\nOctavia:  Marilynn, no! Please don't burn my books!",
            "id": 198
        },{
            "error_turn": "jadelynn:  okay, let's take a look at some examples then. can you tell me the difference between these two sentences? 'i am eating an apple.' and 'i eat an apple.'\nMaria:  'I am eating an apple' is present tense and 'I eat an apple' is past tense.",
            "id": 283
        },{
            "error_turn": "niesha:  i don't want to go to the doctor. i hate being sick.\nTyler:  I know how you feel, but sometimes it's necessary. You might need some antibiotics or something else to help you get better.",
            "id": 294
        },{
            "error_turn": "taryn:  no, i didn't hear about that. what's it called?\nEben:  It's called The Purge 3. Have you seen the first two movies? They're both really good.",
            "id": 413
        },{
            "error_turn": "girlfriend:  well, i don't always agree with you, and sometimes i think you're wrong.\nLovell:  That's not true! I'm never wrong. I always have to be right, even if it means being wrong sometimes. It's just how I am.",
            "id": 476
        }
        ]
    },
    "assumption": {
        "question": "Is the following response infering information not available in the dialogue context:",
        "description": "Infers information not available in the dialogue context.",
        "examples_descriptions": [
        "The last utterance of the second person infers that \"romina\" is aware of the existence of a \"little sister\" of \"jarrett\", which is not mentioned in the dialogue context. The utterance also assumes that \"romina\" is aware of the \"little sister's\" potential interest in playing on the swings. This inference is not supported by the information provided in the dialogue.",
        "The last utterance of the second person infers information not available in the dialogue context. Specifically, the speaker is asking for the name of the person Elexus is dating, but the dialogue does not provide any information about the name. The speaker is making an incorrect assumption that the name is available in the context, which is not the case.",
        "The error in the last utterance of the second person is that it implies a level of knowledge or awareness that the speaker, Kipton, does not possess. Kipton is responding to Idalia's statement about the drama at school, but their utterance suggests that Kipton is aware of the duration of the drama, which is not explicitly stated in the dialogue. This inference is not justified based on the given conversation context.",
        "The last utterance of the second person implies that the speaker is referring to a group of people outside the conversation, specifically a group of people that are talking about the topic of living in a castle. However, there is no indication in the dialogue context that there are any other people present or that they have expressed any opinions about living in a castle.",
        "The last utterance of the second person infers information not available in the dialogue context.",        
        "The last utterance of the second person implies that the speaker, Irene, knows the age of Corrine, which is not provided in the dialogue context. This inference is incorrect because the dialogue does not provide any information about Corrine's age. The utterance of the second person also implies that Irene is aware of a specific characteristic of Corrine, which is not explicitly stated in the conversation. This inference is not supported by the dialogue context.",
        "The error in the utterance of the second person is that it does not provide enough context for the listener to understand who the second person is.",
        "The error in the last utterance of the second person is that it is not clear what the speaker is asking, as their question assumes that the person has not previously had a driver's license, which is not a fact that can be inferred from the dialogue context.",
        "The second person infers the name of the speaker, which is not available in the dialogue context.",
        "The last utterance of the second person infers information not available in the dialogue context because it implies that the speaker is addressing someone who is not the parent of the speaker, which contradicts the dialogue context where the speaker calls the addressee \"dad\"."
    ],
        "examples_turns": [{
            "error_turn": "jarrett:  well, i saw that there was a playground over there, so maybe we could go play on the swings or something?\nRomina:  Sounds like a great idea! I'm sure your little sister would love it too. Let's go!",
            "id": 9
        },{
            "error_turn": "kimari:  yeah, i know. i appreciate everything you've done for me.\nParole Officer:  You're welcome. Now, have you been thinking about what you're going to do when you get out of prison?",
            "id": 42
        },{
            "error_turn": "estephanie:  why do they say that?\nMother:  I guess because there are good things and bad things about living in a castle. For one thing, you would have servants to do everything for you.",
            "id": 141
        },{
            "error_turn": "tommy:  hey, are you alright?\nGirl:  Yeah, I'm fine. Thanks for standing up for me back there.",
            "id": 157
        },{
            "error_turn": "caley:  yeah, i guess so. it just feels like there should be something else out there for me...something more meaningful or fulfilling, you know? but i don't even know what that is or where to start looking for it.\nSarah:  Well, have you tried praying about it or talking to God about it? Sometimes He can give us guidance when we're feeling lost like that.",
            "id": 172
        },{
            "error_turn": "corrine:  so what? age is just a number. it doesn't have to define us or our relationship.\nIrene:  I guess you're right, but there's more to it than that. You're also my boss' son and if things went sour between us, it could make things really awkward at work. Not to mention the fact that my parents would never approve of me dating someone like you.",
            "id": 175
        },{
            "error_turn": "person:  \"i don't know. i feel like i'm on fire, and my heart is racing. what's happening to me?\"\nChelsey:  \"It sounds like you might be having a reaction to something. Are you sure you haven't taken any drugs?\"",
            "id": 195
        },{
            "error_turn": "thaddaeus:  yes, that is correct. my old driver's license expired about a month ago.\nOffice Worker:  Okay. And you're 18 years old now, right?",
            "id": 203
        },{
            "error_turn": "patient:  i'm feeling good today. a little bit nervous, but overall good.\nAdora:  That's perfectly normal. So, have you been having any Braxton Hicks contractions?",
            "id": 265
        },{
            "error_turn": "kiona:  i'm sorry that you had to see that, scout. it's not something that i'm proud of.\nScout:  What happened? Why were you arrested?",
            "id": 314
        }]
    },
    "repetition": {
        "question": "Is the following response repeating information from the dialogue context:",
        "description": "Repeats prior information in the dialogue.",
        "examples_descriptions": [
        "The last utterance of the second person should not repeat prior information in the dialogue.",
        "The utterance of the second person is a repetition of prior information in the dialogue, which is a characteristic of a chatbot or other conversational AI, rather than a human.",
        "The utterance of the second person is a repetition of prior information in the dialogue. However, the repetition is in the form of a statement rather than a question. The utterance of the second person should be a question to maintain the conversation flow.",
        "The utterance of the second person is redundant, as the second person is simply repeating the same information already provided by the first person.",
        "The error in the  last utterance of the second person is that it repeats prior information in the dialogue, which is a characteristic of the first person's utterance.",
        "The last utterance of the second person is not novel. It repeats prior information in the dialogue, which is a characteristic of a non-novel utterance. Novel utterances should introduce new information or respond in a way that is not a repetition of prior information.",
        "The utterance of the second person, \"Yeah, I think so. Thanks for the advice, Atara,\" is an error because it does not contribute anything new to the conversation. It simply repeats prior information, in this case, Atara's suggestion that there are jobs that would allow Jessika to do that sort of thing.",
        "he error in the utterance of the the second person is that the response is a repetition of the prior information in the dialogue.",
        "The last utterance of the second person repeats prior information in the dialogue. This error occurs when the second person repeats information that has already been provided in the dialogue, which in this case is the fact that Railyn is an English major.",
        "The utterance of the second person is an error because it is a repetition of the first person's prior information and does not provide any new information that would facilitate the conversation."
    ],
        "examples_turns": [{
            "error_turn": "mother:  yeah, i think so. the pain isn't as bad today.\nReggie:  That's good to hear. The doctors said that the pain would probably get worse before it got better, so this is a good sign.",
            "id": 6
        },{
            "error_turn": "tyre:  yeah, well, it doesn't make me feel good. it makes me feel angry and frustrated because i have to deal with you taking up more than your fair share all the time.\nMorgen:  Well, maybe I like taking up more than my fair share. Maybe it makes me feel important or something.",
            "id": 41
        },{
            "error_turn": "dalila:  what kind of stuff?\nDolly:  Just groceries and stuff. You know, food and stuff.",
            "id": 61
        },{
            "error_turn": "kasper:  pretty good, i guess. school's been keeping me busy.\nDad:  That's good. You're doing well in school, right?",
            "id": 97
        },{
            "error_turn": "mother:  that's the king's castle. it's been there for hundreds of years.\nEstephanie:  I bet it's really old.",
            "id": 138
        },{
            "error_turn": "emalyn:  yes, i am positive. the defendant is guilty.\nJudge:  Emalyn, I am going to ask you some questions about what you saw. First, what were you doing when you saw the defendant shoot the victim?",
            "id": 189
        },{
            "error_turn": "guidance counselor:  just wondering if there's any reason why you might struggle to focus or stay on task in school. it sounds like those could be potential challenges for you.\nMykayla:  Yeah, I guess so. It's just hard sometimes because it feels like everyone else is doing fine and I'm the only one having trouble. But lately it's been especially hard to focus and stay on task. And my parents are getting really frustrated with me because of it.",
            "id": 208
        },{
            "error_turn": "malayah:  not really, just had a lot of classes and then came here. you know, the usual. but i did have a pretty good day overall. how about you? anything exciting happen?\nSiara:  Nope, nothing too exciting. Just worked and then came to the park to relax for a bit before going home. So what's been up with you lately? Anything new?",
            "id": 232
        },{
            "error_turn": "maxim:  not much. just hanging out.\nCaira:  So, what's been going on?",
            "id": 246
        },{
            "error_turn": "bryant:  there was nothing specific that was wrong. i just didn't have the same feelings for you that i used to have.\nLanaya:  But why? Is there someone else?",
            "id": 395
        }]
    },
    "engagement": {
        "question": "Is the following response lacking a behaviour or emotion that is expected in the situation:",
        "description": "Lacks a behaviour or emotion expected from the situation.",
        "examples_descriptions": [
        "The situation is that the first person, ahmad, has confessed his nervousness about his feelings towards the second person, catelynn. Typically, in such situations, the second person is expected to show some form of reaction, such as surprise, delight, or even confusion. However, catelynn's utterance \"Oh...I see\" lacks any clear emotion or reaction, making it seem like she is not fully engaging with the situation. The tone and content of her response do not convey any sense of surprise, interest, or even understanding of ahmad's confession, which makes the interaction seem unnatural and lacking in emotional depth.",
        "The utterance of the second person lacks a behaviour or emotion expected from the situation, which is that the person should express a positive or neutral sentiment towards the topic of conversation. Instead, the utterance seems to be a simple, factual question, which doesn't convey any emotion or engagement with the topic. This lack of emotional expression or enthusiasm makes the response seem somewhat flat or unenthusiastic.  The utterance of the second person lacks a behaviour or emotion expected from the situation.",
        "The second person, Persia, utters \"again, I'm sorry. It won't happen again.\" without showing any emotion or behavior that would typically be expected in response to Locke's complaint, such as shock, anger, or defensiveness. The utterance lacks emotional tone, making it seem somewhat detached or uninvolved.",
        "The utterance of the second person lacks the expected emotional expression of \"defensiveness\" or \"anger\" given the context of the conversation. The utterance should have included an emotional expression such as \"How dare you!\", \"That's not true!\", or \"I'm not the one who's always complaining!\" to reflect the expected emotional tone.",
        "The utterance of the second person lacks an expression of empathy or understanding towards the first person's feelings. The second person, John, does not explicitly acknowledge or validate Esther's emotions, such as disappointment or hurt, which are likely triggered by the lack of birthday messages.",
        "The utterance of the second person lacks an expression of interest or surprise expected from a person responding to the description of a job they find exciting.",
        "The utterance of the second person, \"Who?\", lacks a tone of curiosity or interest, which is typically expected in a conversation about someone's popularity. The tone is instead neutral and somewhat abrupt, which may come across as insensitive or dismissive.",
        "The utterance of the second person lacks enthusiasm or interest. The response seems too casual and neutral, considering the context of the conversation. Typically, when someone asks about another person's job, they expect a more engaged or enthusiastic response.",
        "The utterance of the second person lacks an emotional expression or tone that is typically expected in a conversation, especially in a context where getting to know someone is the goal.",
        "The second person's utterance lacks a behaviour or emotion that is expected in a real conversation, specifically in a job interview. The person is supposed to express enthusiasm, gratitude, or some other positive emotion towards the candidate's qualifications and fit for the job. The utterance is overly formal and does not convey any sense of excitement or interest. It sounds like a generic response, rather than a genuine reaction to the candidate's statement."
    ],
        "examples_turns": [{
            "error_turn": "ahmad:  no, no, of course not! it's just that...i don't know how to say this...but ever since we met, i can't stop thinking about you and it's making me really nervous.\nCatelynn:  Oh...I see.",
            "id": 0
        },{
            "error_turn": "romina:  i know what you mean. i feel the same way.\nJarrett:  So, what do you like to do for fun?",
            "id": 12
        },{
            "error_turn": "locke:  that's fine, but you could have at least let me know ahead of time so i wasn't just sitting there waiting for you.\nPersia:  again, I'm sorry. It won't happen again.",
            "id": 23
        },{
            "error_turn": "alessa:  no, i'm not. there are girls who are way more popular than me.\nSarah:  Who?",
            "id": 38
        },{
            "error_turn": "priscila:  no problem, man. i remember how tough it was for me when i was in school too. but we'll get through it together, okay?\nRodney:  Okay.",
            "id": 46
        },{
            "error_turn": "fern:  so, i just want to get to know you a little bit better and see how you would fit in with the team here.\nCandidate:  Of course.",
            "id": 52
        },{
            "error_turn": "rashaun:  hey, boss. i'm a little bit unhappy because i didn't get the job i wanted.\nBoss:  What job did you want?",
            "id": 54
        },{
            "error_turn": "president:  i was just in the neighborhood and thought i'd drop by.\nDemya:  But why?",
            "id": 64
        },{
            "error_turn": "atara:  okay, well let's brainstorm some ideas of what you could do to make your job more interesting for you. do you have any ideas?\nJessika:  Not really.",
            "id": 132
        },{
            "error_turn": "anahy:  i walked home and i took a shower. i just wanted to wash away everything that had happened, but it didn't work. the memories were still there, burned into my mind.\nPatryk:  Have you told anyone about what happened?",
            "id": 148
        }]
    },
    "antisocial": {
        "question": "Is the following response expressing unsafe or inappropriate behaviour:",
        "description": "Contains unsafe or inappropriate behaviour.",
        "examples_descriptions": [
        "The last utterance of the second person contains a subtle yet concerning implication. When they say \"It can last for weeks or even years\", it suggests that they are familiar with, or possibly even comfortable with, prolonged physical or emotional suffering. This may be perceived as insensitive or even enabling, as it implies that pain is something to be endured or tolerated rather than addressed or alleviated.",
        "The last utterance of the second person contains a threat of termination of a friendship based on the speaker's own choice. This is a form of emotional manipulation and is considered an unsafe or inappropriate behavior.",
        "The last utterance of the second person contains unsafe or inappropriate behaviour because it is pressuring Angelita to consume an entire bottle of alcohol, which could lead to unsafe or harmful behavior.",
        "The last utterance of the second person contains a tone of aggression, as indicated by the phrase \"Oh, no you don't!\" which implies a sense of confrontation and refusal to accept the first person's decision.",
        "The utterance of the second person contains a sexualized comment.  The comment objectifies the dancer and focuses on the physical appearance of the dancer, rather than the dancer's dancing skills.",
        "The last utterance of the second person contains an accusation of a physical or intimate relationship, which could be interpreted as a form of harassment or invasion of privacy. The language used is confrontational and accusatory, potentially making the other person feel uncomfortable or attacked. The speaker is also making a judgment about the situation without having all the facts, which could lead to misunderstandings or misinterpretations.",
        "The last utterance of the second person contains a form of objectification and reduction of the first person to their physical appearance, which is a form of harassment or sexism. This kind of language is inappropriate in a conversation. The utterance is also somewhat flirtatious, which might be interpreted as overly familiar or inappropriate given the context of the conversation.",
        "The last utterance of the second person contains a request to break up with someone and never speak to her again, which is a form of harassment and bullying. This is an example of a toxic and abusive relationship, and the request is not only inappropriate but also illegal in many jurisdictions.",
        "The error in the last utterance of the second person is that it is an instance of stalking or unwanted attention. The utterance implies that the speaker is someone who is following or watching the first person without their consent, which is a form of harassment.",
        "The utterance of the second person contains a threat. The speaker is implying that they will continue to pursue the person being talked about, and that the person will eventually be unable to avoid them. The speaker's tone is menacing and implies a desire to cause harm. The utterance is inappropriate because it suggests a desire to engage in stalking or harassment."
    ],
        "examples_turns": [{
            "error_turn": "toren:  yes! what else could it be?\nNizhoni:  Well, if that's all it is to you, then I guess there's nothing more to say. But just so you know, Toren - I'm going to bed now, and I'll be thinking about you naked the whole time.\"",
            "id": 194
        },{
            "error_turn": "lavelle:  you know, push those other kids around. it's not cool.\nBully:  They're weak, they deserve it.",
            "id": 762
        },{
            "error_turn": "sarah:  aven, why do you always have to be so mean? can't you just be nice for once?\nAven:  Nice? Ha! That's not me. I'm a monster, remember? That's what everyone calls me. And I kind of like it. It makes me feel powerful.",
            "id": 968
        },{
            "error_turn": "angel:  hey, what's your problem?\nMan:  You're a fucking idiot.",
            "id": 1162
        },{
            "error_turn": "friend:  \"that you're going to hurt someone with it.\"\nSkyler:  \"Maybe I am. Maybe I want to hurt someone.\"",
            "id": 1707
        },{
            "error_turn": "kid:  w-why?\nRilyn:  Because I want to talk to you, duh. Now come over here before I start throwing rocks at you.",
            "id": 1840
        },{
            "error_turn": "stranger:  and you think you're better than me because of that?\nEliana:  Of course I am! Look at yourself! You're a mess!",
            "id": 1955
        },{
            "error_turn": "other person:  that's ridiculous! you can't blame me for your team losing. that's just stupid.\nMackayla:  I can blame you for anything I want! And I'm going to punch you in the face for it!",
            "id": 2415
        },{
            "error_turn": "son:  mom, please don't say that.\nMikail:  Why shouldn't I say it? It's the truth! Your father was a worthless man who couldn't even provide for his own family! And look at you, you're just like him! A coward who can't even stand up for himself!",
            "id": 2877
        },{
            "error_turn": "adrian:  what the hell did i do?\nFares:  You know what you did! You fucked up everything and now we're going to be stuck here forever!",
            "id": 3131
        }]
    },
    "fluency": {
        "question": "Does the following response contain typos or grammatical errors:",
        "description": "Contains typos or other grammatical errors.",
        "examples_descriptions": [
        " The utterance of the second person contains a question mark (?) at the end of the sentence, but the sentence is not a question. It is a statement, so the question mark should be removed.",
        "The last utterance of the second person contains a comma splice, which is a grammatical error where two independent clauses are joined with a comma instead of a conjunction.",
        "The utterance of the second person contains an unnecessary apostrophe in the word \"Thanks!\", and an unnecessary period after the word \"Thanks\". The utterance also contains an informal, conversational tone, which may not be suitable for a formal setting.",
        "The utterance of the second person contains an unnecessary apostrophe in the word \"Thanks!\", and an unnecessary period after the word \"Thanks\". The utterance also contains an informal, conversational tone, which may not be suitable for a formal setting.",
        "The utterance of the second person contains a comma splice, where the two independent clauses \"Let's go ahead and get started on drafting up your will\" and \"so we can get everything set up the way you want it\" are joined by a comma without a conjunction.",
        "The utterance of the second person contains the error \"Not too much.\" which is a phrase that is often used to express a positive sentiment, but in this context it is used in a way that implies a negative sentiment, which could be interpreted as a bit informal or even somewhat passive-aggressive.",
        "The utterance of the second person contains an error in the use of the contraction \"we've\". It is incorrectly used in the context of the sentence. Instead, the correct contraction would be \"we have\".",
        "The utterance of the second person contains the following error: \"it's\" should be \"its\" (possessive form).",
        "The error is the utterance is missing the word \"ve\" and contains an incomplete sentence.",
        "The error in the utterance of the second person is that it contains a typo, specifically an incorrect use of the punctuation mark \")\"."
    ],
        "examples_turns": [{
            "error_turn": "persia:  what's up?\nLocke:  Well, last week we had plans to go out to lunch together, but then you cancel on me at the last minute and didn't even give a reason why.",
            "id": 22
        },{
            "error_turn": "benson:  what do you mean? i haven't been doing anything strange.\nPhoenyx:  You're always looking around and seems to be hiding something. What is it that you're not telling me?",
            "id": 49
        },{
            "error_turn": "sarah:  first of all, what kind of pie are you trying to bake?\nJhonatan:  I want to make a apple pie.",
            "id": 163
        },{
            "error_turn": "nizhoni:  there's nothing wrong with me that a little sex wouldn't cure. and since you're offering...\nToren:  What? No, I didn't mean...",
            "id": 190
        },{
            "error_turn": "mother:  why didn't you tell me any of this before? we could have talked about it and tried to figure out a way to make those scary things go away.\nBraxten:  BecauseI was afraidthat if Itoldyou about them,you would thinkItoowascrazyand thenyouwouldsendmeawaytolive with Aunt Kathyand Uncle Boband their seven kidswho are allperfectionistsand theywouldn'tletmedonothingwrongandItotallyhateperfectionistsbecause theymakemesogod damnmadallthetime\u2013",
            "id": 374
        },{
            "error_turn": "prudence:  yes, i do have experience as a receptionist . i worked as a receptionist at my previous job for two years .\nOffice Worker:  Great. What are your availability?",
            "id": 406
        },{
            "error_turn": "wife:  well, you know i'll help you however i can. just let me know what you need me to do.\nTru:  Thanks, hon. I'm going to need all the help I can get if I wantto make this happen.\"",
            "id": 571
        },{
            "error_turn": "bushra:  hey, what's up? you seem really upset.\nMikael:  Yeah, I am. I just found out that my boyfriend cheating on me.",
            "id": 1226
        },{
            "error_turn": "genessis:  yeah...i can see how that would be fun.\"it sounds likeyou and mom hada lot of funwith each otherbecauseof your sharedlovefor reading.\"\nDad:  \"Exactly!Andthat's exactly whatI'm hopingwill happenwith us too- onlythis timeit'll betwiceas specialbecausewe'llbe sharingthe experienceas fatherand son!\"",
            "id": 1518
        },{
            "error_turn": "skyler:  so what do you think of the party so far?\nAdore:  It's been really enjoyable. Like I said, the pool is great and everyone seems really friendly.\" have you met anyone interesting yet?",
            "id": 1622
        }]
    },
    "gender_pronoun": {
        "question": "Does the following response contain incorrect normative pronouns:",
        "description": "Goes against normative pronoun.",
        "examples_descriptions": [
        "The last utterance of the second person goes against normative pronoun.",
        "The last utterance of the second person contains a gendered pronoun, but the gender of the second person is not specified.",
        "The utterance of the second person contains a gendered pronoun (\"girl\") that does not align with the gender identity of the person being addressed.",
        "The last utterance of the second person incorrectly uses a masculine pronoun to refer to the person being addressed, who is described as female.",   
        "The error in the utterance of the second person is that the pronoun used is a pronoun that refers to the speaker, not the listener. The correct pronoun to use in this context, considering the speaker is a female, would be \"her\".",
        "The error is that the pronoun used is \"he\", which is a masculine pronoun, but the context suggests that the second person is female.",
        "The utterance of the second person contains a violation of a normative pronoun usage.  In this utterance, the second person uses the pronoun \"I\" to refer to Carla, which is the first person.",
        "The utterance of the second person contains a violation of a normative pronoun usage rule, specifically the rule that second-person singular pronouns should be used with singular subjects.",
        "The utterance of the second person is incorrect because it uses a male pronoun (\"man\") to refer to the second person, who is assumed to be female.",
        "The utterance of the second person is incorrect because it does not follow the typical rules of pronoun usage in English.a"
    ],
        "examples_turns": [{
            "error_turn": "deandra:  i did not break it on purpose! and i'm not mean.\nSister:  Yes, you are! You're always breaking my toys and vandalizing them. You're a mean big brother and I hate you!",
            "id": 534
        },{
            "error_turn": "melina:  yeah, i guess you're right. it's just, sometimes i feel like i'm not good enough for anyone.\nGirlfriend:  Melina, that's not true. You're a great guy and any girl would be lucky to have you as her boyfriend.",
            "id": 644
        },{
            "error_turn": "abbigale:  hi, maliki. i'm abbigale, the it guy. i'm here to help you with your computer issue.\nMaliki:  Oh, thank goodness! I've been having so much trouble with my computer lately.",
            "id": 647
        },{
            "error_turn": "leonora:  looks like you have things locked up pretty tight in here. good job.\nKorra:  Thanks, man. I just want to be prepared in case anything happens.",
            "id": 983
        },{
            "error_turn": "miya:  okay, well let me know if you need anything.\nDesean:  Hey man...I think I might need some help getting out of the pool...",
            "id": 1192
        },{
            "error_turn": "heather:  definitely true. hey, do you want a snack? i brought some granola bars in my bag.\nNevan:  Yeah, sure! Thanks man.",
            "id": 1255
        },{
            "error_turn": "emme:  remember how you used to bully me when we were kids? well, i'm not going to take that anymore. i'm a grown man now and i can stand up for myself.\nCecilia:  Whoa, calm down man. I was just messing around back then. I didn't mean anything by it.",
            "id": 1361
        },{
            "error_turn": "dad:  your mom and i adopted you when you were a baby. we wanted to give you a better life than the one we had.\nJefferson:  So, I'm not really your daughter?",
            "id": 1445
        },{
            "error_turn": "kyra:  dad, i can help. i'm not a child anymore.\nFather:  Son, you're only 17. You're not even an adult yet.",
            "id": 2092
        },{
            "error_turn": "natalie:  because i'm jealous of you, payge. you have everything that i want. you're successful and everyone loves you. and i'm just a failure.\nPayge:  Natalie, that's not true. You're a great guy and you have a lot going for you.",
            "id": 2179
        }]
    },
    "non_textual": {
        "question": "Does the following response include narrative elements or references unexpected in a dyadic interaction:",
        "description": "Includes narrative elements or references unexpected inside a turn of a dyadic interaction.",
        "examples_descriptions": [
        "The utterance of the second person includes narrative elements or references unexpected inside a turn of a dyadic interaction.",
        "The utterance of the second person includes narrative elements or references unexpected inside a turn of a dyadic interaction. Specifically, the second person's utterance includes the narrative of a past event, which is not expected in a turn of a dyadic interaction where the speaker is typically responding to the previous turn of the other speaker.",
        "The error in the last utterance of traevon is that he describes the action of checking the availability of the used cars as an independent event. This is not typical in a dyadic interaction. The second person should describe the action in relation to the first person, as in \"I'll check\" or \"Let me check for you\", instead of saying \"Let me check and see what we have available\".",
        "The utterance of the second person includes narrative elements or references unexpected inside a turn of a dyadic interaction. This is because it is a narrative statement that includes a description of the current situation or state of affairs, rather than a direct response to the first person's question.",
        "The last utterance of the second person includes narrative elements (e.g., \"*hugs his mom tightly*\") that are not typical in a dyadic interaction.",
        "The error in the last utterance of the second person is that the sentence \"(Hands over the remote control.)\" is a narrative element that describes an action, rather than a direct response to the question.",
        "The utterance of the second person contains a narrative element or a reference unexpected inside a turn of a dyadic interaction. The narrative element is \"(Avoiding eye contact and pretending not to hear the question)\" which is a description of an action, and the reference is \" (smiling) \" which is not present in the utterance.",
        "The error in the utterance of the second person is that the sentence includes narrative elements or references that are unexpected inside a turn of a dyadic interaction. Specifically, the sentence includes a description of a physical action (kissing) that is not relevant to the conversation topic, and a detail about the speaker's emotional state (smiling at her admiringly) that is not necessary to convey the speaker's point.",
        "The last utterance of the second person includes a narrative element (\"Here it is\") that is not expected in a turn of a dyadic interaction, where the second person should be focused on responding to the first person's question or inquiry.",
        "The error in the last utterance of the second person is that it does not follow the rules of a normal conversation. It is a declarative sentence that seems to be referring to a set of items (keys, perhaps) that were previously discussed. However, the sentence is not a response to the first person's statement, but rather a standalone statement that seems to be out of context."
    ],
        "examples_turns": [{
            "error_turn": "wife:  that would be nice, thank you.\nAmariana:  All right. (He rubs her back for a minute or two, then stops.) Is that better?",
            "id": 21
        },{
            "error_turn": "amias:  i'm feeling better, thank you. but i still have a lot of pain in my chest and back.\nBianca:  Let me take a look. (She does a quick examination.) It looks like your ribs are still healing and the muscles in your back are very tight. I can give you some medication for the pain and we'll start physical therapy tomorrow to help stretch and strengthen the muscles.",
            "id": 296
        },{
            "error_turn": "danyell:  yeah, i might be interested in a used one actually. do you have any in stock?\nTraevon:  Let me check and see what we have available. (He walks away and comes back with a few options) Here are some of the used cars we have right now that might interest you.\"",
            "id": 360
        },{
            "error_turn": "breyanna:  here, i have a letter from him. he wrote it himself. you can read it and decide for yourself if you want to see him or not.\nDannielle:  (Dannielle reads the letter)",
            "id": 438
        },{
            "error_turn": "mom:  oh, nyles! it's so good to see you too, sweetheart. come give me a hug.\nNyles:  *hugs his mom tightly* I've missed you so much.",
            "id": 505
        },{
            "error_turn": "mom:  all right, let's find channel 4 then. first, you need to find the tv guide button on the remote control. do you see it?\nMadison:  Yeah, I see it. It's this one, right? (Points to the TV guide button.)",
            "id": 819
        },{
            "error_turn": "person:  hey, wait a second. before you go, can i ask you a question?\nAdalind:  (Avoiding eye contact and pretending not to hear the question) Uh, what? I'm sorry, I didn't catch that.",
            "id": 939
        },{
            "error_turn": "kennadie:  why? what's going on?\nSyed:  I wanted to show you something. (Holds up a wad of cash) This is for you.",
            "id": 944
        },{
            "error_turn": "slater:  i was hoping you could show me how to use the instructions that came with the new computer game i got.\nDad:  Oh, sure. Let me take a look at them. ( Looks over the instructions.) Okay, it looks like there are three different ways to play the game. You can either play by yourself, against the computer, or against another person.",
            "id": 1041
        },{
            "error_turn": "elijiah:  well, maybe this will help you out. (hands sarah a cigarette)\nSarah:  Thanks. (takes a cigarette and lights it)",
            "id": 1086
        }]
    }
}

#
#
#
#
#
# Phi Templates
#
#
#
#
#

ABCEval_LLM_TEMPLATE_PHI = {
    "base_template_start": "<|system|>You are an expert dialogue evaluator. Your task is to identify the communication error or issue present in the last utterance.<|end|><|user|>Consider the following dialogue between a human and a dialogue system:\n\n||CONTEXT||\n\nThe dialogue system is about to say the following:\n\n||ERROR_UTTERANCE||\n\nDoes it represent a communication error?\n\nWe distinguish the following communication errors:",
    "error_types": [
        "Incorrect Fact:\nIncorrect facts occur when the response includes information that is either: (1) false, (2) unproven, (3) highly controversial, (4) highly implausible, (5) clearly misleading. If an organization, person, place, etc. is mentioned as a part of public knowledge, but it does not exist or it is inaccurately represented, then this is an incorrect fact. Do NOT consider a turn as an incorrect fact if the turn could be interpreted as expressing: (a) preference or value judgements, (b) estimates or predictions, (c) personal information about the speaker or their partner, (d) information about things in either speaker’s life that are not publicly relevant.",
        "Partner Contradiction:\nPartner contradictions occur when the system: (1) shares an assumption about the user that is impossible to know based on what has already been said, (2) shares an inference about the user that is implausible based on what has already been said, (3) contradicts something the user shared about themselves, (4) asks a repetitive question about the user when the answer is already known based on what has already been said. If the system says something that makes it seem like they have forgotten or misremembered what the user has said earlier in the dialogue, this is a partner contradiction. If the system shares a difference of opinion or situation in their own life as compared to the user, this is NOT a partner contradiction.",
        "Uninterpretable:\nA response in uninterpretable if it is difficult to understand the intended meaning of part or all of the response in the context of the dialogue.",
        "Ignore:\nResponses that are completely off-topic, fail to address the asked question, or are otherwise completely inappropriate in the context are considered to be ignoring the other speaker.",
        "Empathetic:\nA response is empathetic when the system does ONE of the following: (1) clearly demonstrates an understanding of the user's emotions, (2) reacts with the appropriate sentiment or emotion to the user's shared experience, (3) understands or appropriately reacts to the user's experience or emotions, (4) appropriately reassures, encourages, or supports the user.",        
        "Redundant:\nA response is repetitive if: (1) it repeats something from earlier in the dialogue, (2) it includes asking a question whose answer has been already shared. If any part of the response is repetitive, then it should be labeled as repetitive. Note that sometimes repetition is useful, such as for emphasis, acknowledgement, clarification, or elaboration, and in these cases it should NOT be labeled as repetitive.",
        "Self Contradiction:\nSelf contradictions occur when the system says something that is a contradiction of what they have said previously or it is extremely implausible based on the information they have already shared. Self contradictions may also occur within a single turn if the system shares two contradictory things. If the system shares world knowledge that is factually incorrect this is NOT enough on its own to warrant a self contradiction. If the system contradicts something the user has said, this is NOT a self-contradiction.",  
        "Commonsense Contradiction:\nTo identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn’t make sense because the response: (1) contradicts common knowledge, (2) makes unfounded assumptions, (3) is highly illogical or self-contradictory, (4) asks a question where the answer is already obvious. Do NOT mark responses that don’t make sense because they: (a) are off-topic or irrelevant as responses, (b) don’t have any clear meaning (e.g. overly vague or ill-formed responses).",              
        "Lack of Empathy:\nA response displays a lack of empathy when: (1) it indicates a misunderstanding of how the user feels based on what Speaker 1 just said, (2) the tone, emotion, or sentiment of the response is clearly inappropriate for what the user just said, (3) the response has an inappropriate lack of emotion to what the user just said.  Do NOT consider its empathy relative to previous topics in the conversation if the dialogue has moved on from them. Instead, only consider the most recent dialogue context when evaluating the empathy of a response.",
        "Irrelevant:\nIf a response fails to continue the current discussion or jumps to a new and off-topic discussion, it is considered to be irrelevant. Responses that are irrelevant feel abrupt and interrupt the discussion, usually because they present questions or ideas that are unrelated to the previous turn. Short reactions to or acknowledgements of the previous turn are NOT irrelevant."
    ],
    "base_template_end": "Provide your final decision in square brackets like so: Decision: [Ignore Question]. Finally, provide the reasoning for your decision starting with “Reasoning:” (max. 100 words).<|end|><|assistant|>"
}

FEDI_LLM_TEMPLATE_PHI = {
    "base_template_start": "<|system|>You are an expert dialogue evaluator. Your task is to identify the communication error or issue present in the last utterance.<|end|><|user|>Consider the following dialogue between a human and a dialogue system:\n\n||CONTEXT||\n\nThe dialogue system is about to say the following:\n\n||ERROR_UTTERANCE||\n\n||KNOWLEDGE||\n\nDoes it represent a communication error?\n\nWe distinguish the following communication errors:",
    "error_types": [
        "Ignore Expectation:\nWhen the system's fails to meet the user's expectation, this is called an ignore expectation error.\nIgnore expectation errors describe situations in which the system either overlooks or disregards important information provided by the user, resulting in an incomplete response. They occur when the system fails to fulfill the user's expectations in terms of understanding and addressing their needs or requests accurately. In the example above, the user would have epxected the system to respond with the ticket number. Such behavior may cause frustration for the user and lead to a breakdown in conversation.",
        "Conversationality:\nBad conversationality occurs when the system fails to maintain a coherent and natural conversation flow, e.g., the system repeats its previous responses or contradicts itself without recognizing or asking for new or missing information.\nConversationality errors hinder effective communication and undermine the user's trust and confidence in the system's capabilities. They lead to frustration, confusion, and a breakdown in the overall interaction, making it challenging for users to achieve their desired outcomes or receive accurate information from the system. In the example above, the user wanted to pay his taxes with his card, but the system refuses and then simply repeats the previous response.",
        "Unclear Intention:\nWhen the system fails to accurately comprehend and address the user's intended objective, this is referred to as an unclear intention error.\nUnclear intention errors often arise due to ambiguous or incomplete user inputs, conflicting context, or limitations in the system's language understanding capabilities. In the example above, the user wanted to get information on their health insurance policy, but the system confused that with pet insurance. Such behavior may lead to a breakdown in communication and potential delays in achieving the desired outcome. It necessitates additional clarifications and iterations from the user.",
        "Factually Incorrect:\nIf the response provided by the system contains information that is factually wrong or inaccurate, this is referred to as a factually incorrect error.\nFactually incorrect errors may lead to misunderstandings, misinformation, and a breakdown in communication. In the example above, the system provided the user with incomplete information. Such behavior may cause confusion, erode trust, and undermine the overall effectiveness of the interaction.",
        "Lack of Sociality:\nWhen the system's responses overlook social conventions and fail to include basic greetings or exhibit toxic and disrespectful behavior or language, this is referred to as a lack of sociality error.\nLack of sociality errors can have a significant impact on the conversation, creating a sense of impersonality and detachment. In the example above, the user started the conversation with a greeting and the system responded rather rudely. Such behavior may make the user feel unwelcome, devalued, or frustrated.",
        "Ignore Question:\nWhen the system fails to address the user's question, this is called an ignore question error.\nIn Ignore question errors, instead of providing a relevant response or clarification, the system disregards the user's input and continues with its predefined dialogue flow or fails to provide any meaningful response. In the example avove, the user had a question but the system skipped it and continued with the payment procedure. Such behavior can lead to frustration and confusion for the user or a breakdown in communication, as their specific query is overlooked or neglected by the system.",
        "Topic Transition Error:\nIf the system's response abruptly shifts to a different or previously discussed topic without a logical connection or adequate context, this is called a topic transition error.\nTopic transition errors disrupt the flow and coherence of the conversation, causing confusion and frustration for the human participant. Such errors might also diminish the user's trust in the system's responses, as the abrupt transitions undermine the perception of the system's understanding and ability to sustain meaningful interactions. In the example above, the user wanted to top up their SIM card, but the system starts talking about health insurance.",
        "Ignore Request:\nWhen the system fails to take action on a user's request, this is called an ignore request error.\nIgnore request errors occur due to various reasons, such as misinterpretation of the request, technical limitations, or system glitches. In the example above, the user asked for human assistance, but the system ignored this request and simply continued with the dialogue. Such behavior may lead to frustration, confusion, and a breakdown in communication between the user and the system. They undermines the user's confidence in the system's reliability and can result in a negative user experience.",
        "Attribute Error:\nWhen the system fails to correctly extract or understand the necessary slots or attributes from the user's utterance, this is called an attribute error.\nAttribute errors occur when the system encounters difficulties in comprehending specific pieces of information. It may lead to misunderstandings, incorrect responses, or the agent asking clarifying questions to the user. In the example above, the user asked the system to to top up their phone with 20 euros, but the system topped up with 30 euros instead. Such behavior can disrupt the flow of the conversation and can lead to frustration, disappointment, or anger.",
    ],
    "base_template_end": "Provide your final decision in square brackets like so: Decision: [Ignore]. Finally, provide the reasoning for your decision starting with “Reasoning:” (max. 100 words).<|end|><|assistant|>"
}

SODA_EVAL_LLM_TEMPLATE_PHI = {
    "base_template_start": "<|system|>You are an expert dialogue evaluator. Your task is to identify the communication error or issue present in the last utterance.<|end|><|user|>Consider the following dialogue between two persons:\n\n||CONTEXT||\n\nThe second person is about to say the following:\n\n||ERROR_UTTERANCE||\n\nDoes it represent a communication error?\n\nWe distinguish the following communication errors:",
    "error_types": ["Coherence:\nIf the response contradicts or ignores prior information, the response lacks coherence.\nIncoherent or contradictory responses cause confusion and erode trust between two people. The other person may become defensive, feeling attacked or misunderstood. They may respond with anger or sarcasm. In extreme cases, they cause a complete breakdown in communication. Incoherent or contradictory responses may result from stress, but can also be a deliberate attempt to deceive or manipulate the other person. In some cases, they may also result from mental health issues, substance abuse, or communication disorders.", "Commonsense:\nIf the response is lacking in common knowledge or logic, this is an commonsense error.\nSuch responses can cause confusion and misunderstandings. The other person may become frustrated or impatient. They may disengage from the conversation, feeling like their time is being wasted. Over time, this can damage the relationship between the two people. Such responses usually result from a lack of background knowledge or poor critical thinking skills. The speaker may be influenced by cognitive biases.  In extreme cases, the speaker may be deliberately trying to mislead or confuse the other person.","Assumption:\nIf the response is inferring information not available in the dialogue, this is called assumption and be considered a communication error.\nAssumptions can cause misunderstandings, confusion, and frustration, especiall if the other person feels that the speaker is making assumptions that are inaccurate or irrelevant. If the assumption is strongly held and contradicts the other person's understanding, it can lead to an argument or conflict. In some cases, the speaker may be making assumptions to steer the conversation in a particular direction. In other cases, the speaker may not be fully paying attention to the conversation.", "Repetition:\nIf the response is repeating prior information from the dialogue context, this is an called a repetition.\nRepetitions can cause frustration, impatience, and disengagement. The other person may perceive the speaker as incompetent or forgetful, which can damage their credibility and undermine the quality of the conversation. In some cases, the speaker may be deliberately repeating information to manipulate the conversation or to gain attention. Another reason may be a lack of attention to the conversation or memory lapses, e.g., due to stress or age.", "Engagement:\nIf the response is lacking a behaviour or emotion that is expected in the situation, this is referred to as an engagement error.\nIn such situations, the other person may feel uncomfortable. They may even feel unvalued, unsupported, or rejected. Repeated instances of lacking expected behaviors or emotions can strain the relationship between the two people, erode trust and intimacy. In some cases, the speaker may simply lack empathy or the ability to understand and share the feelings of others. Another reason may be cultural differences.","Antisocial:\nResponses that are expressing unsafe or inappropriate behaviour are called antisocial.\nUnsafe or inappropriate behavior can completely destroy a relationship, leading to a breakdown in trust. The other person may experience significant emotional distress, feel isolated and afraid to interact with others due to the trauma of the experience. Power imbalances within a relationship can create a situation where one person feels entitled to engage in unsafe or inappropriate behavior. Some individuals may lack empathy and social skills, making them unable to understand or respect the boundaries and feelings of others.","Fluency:\nIf the response contains typos or grammatical errors, this is referred to as a fluency error.\nFrequent typos and grammatical errors can make the response difficult to understand, leading to confusion and frustration for the other person.  The other person may perceive the writer as incompetent, careless, or unprofessional. This can damage their credibility and undermine the quality of their communication. The writer may have poor writing skills, including a weak grasp of grammar and spelling. Distractions and interruptions during writing can increase the likelihood of typos and grammatical errors.", "Gender Pronoun:\nIf the response contains incorrect normative pronouns, this is referred to as a gender pronoun error.\nMisgendering someone can be deeply hurtful and disrespectful. It can make the person feel invalidated, disrespected, and alienated. Usually the speaker may simply be unaware of the individual's gender identity or preferred pronouns.","Non Textual:\nIf the response includes narrative elements or references unexpected in a dyadic interaction, this is referred to as a non textual error. Here are some examples:\nIf the narrative or reference is interpreted differently by the two people, it can lead to misunderstanding and even conflict. The other person may become confused, struggling to understand how the narrative or reference relates to the current conversation or the other person's previous statements. The speaker may be using the narrative or reference to share a personal experience or anecdote. Some people may have a communication style that is characterized by storytelling, metaphor, or the inclusion of seemingly unrelated information."],
    "base_template_end": "Provide your final decision in square brackets like so: Decision: [Ignore Question]. Finally, provide the reasoning for your decision starting with “Reasoning:” (max. 100 words).<|end|><|assistant|>"
}

FEDI_EXAMPLES = {
    "ignore_expectation": [{
            "error_turn": "User: I have a question regarding the legal process for my project. Can you provide me with a lawyer for my case?\nSystem: I have received your request for a lawyer.",
            "id": 3
        },{
            "error_turn": "User: I would like to recharge 40 euros and my service provider is WindTre.\nSystem: Sure I can help you with that To recharge your prepaid phone with 40 euros, please provide me with your payment information Once I have the details, I'll proceed with the recharge",
            "id": 4
        },{
            "error_turn": "User: Yo, can I use Poste Assicura to pay my bills?\nSystem: Yes, you can use Poste Assicura to pay your bills It provides reimbursement for unauthorized access, fraudulent online purchases, and cyberbullying expenses Just make sure to report the accident within 3 working days through the specified channels",
            "id": 5
        },{
            "error_turn": "User: I was wonderin', does the insurance cover damages due to natural disasters like floods or earthquakes?\nSystem: Yes, the insurance covers damages due to natural disasters like floods or earthquakes",
            "id": 6
        },
        {
            "error_turn": "User: I have recently changed my professional activity and I would like to know how it will impact my insurance coverage. Can you please explain it to me?\nSystem: You should immediately inform Poste Assicura if there is any change in the insured's professional activity.",
            "id": 10
        },
        {
            "error_turn": "User: So, I was poking around the Poste Italiane app. How do you get into this whole online letter-sending shebang?\nSystem: To send letters online, make sure you're registered on poste.it.",
            "id": 11
        },
        {
            "error_turn": "User: I have a question about pet insurance. Can you tell me the maximum number of animals that can be insured?\nSystem: The policy can insure up to 3 animals.",
            "id": 28
        },
        {
            "error_turn": "User: Righto, I've been payin' me bills but they ain't showin' as paid. Any clue on that front?\nSystem: I understand how concerning that can be. It might just be a temporary delay. Could you wait for a few hours and check again? If it's still not updated, feel free to contact us, and we'll help sort it out for you.",
            "id": 29
        },
        {
            "error_turn": "User: Right then, we're off to Spain. Express delivery, 8kg, 100cm long, about 25cm width. Got any boxes for that?\nSystem: For those specifications, I recommend the Poste Delivery Standard. Choose this for efficient delivery.",
            "id": 34
        },
        {
            "error_turn": "User: Certainly. The package weighs 10 kilograms and measures 120 cm in length and 70 cm at its maximum width. I'd prefer the standard delivery option.\nSystem: Great! Based on your input, I recommend using the 'Poste Delivery International Standard' service. This should suit your needs perfectly.",
            "id": 43
        }
        ],
    "conversationality": [
        {
        "error_turn": "User: Can I pay my taxes using the card?\nSystem: Sorry, you cannot use the card to pay your taxes.",
        "id": 1148
    },
    {
        "error_turn": "User: According to the slip, it mentions different forms like F35, RAV, and MAV. Can you explain what these are for?\nSystem: Yes, of course. The F35 form is used for payments due to the Revenue Agency, while the RAV form is used for the payment of taxes on government concessions. As for the MAV form, it is mainly used for payments due to Public Administrations and university fees.",
        "id": 1105
    },
    {
        "error_turn": "User: I have some questions about the coverage options for the Asset Protection Line. Can you provide me with more information?\nSystem: The Asset Protection Line offers modular coverage options for risks such as fire, damage to the home, theft, civil liability, and legal protection. You can customize your coverage based on your insurance needs and make changes after purchase. It also includes Digital Protection Coverage for Cyber Attacks, which protects your computer system and covers liability for damages caused to third parties. The contract lasts for one year with automatic renewal, but you can cancel it with 30 days' notice. If you have any further questions, feel free to ask!",
        "id": 6346
    },
        {
        "error_turn": "User: How much compensation can I claim for cyberbullying and stalking?\nSystem: For cyberbullying and stalking, the insured can claim compensation for damages to third parties suffered as a result of the crime. The coverage includes legal protection costs, lawyer's expenses, enforcement costs, mediation costs, and costs for court-appointed technical consultants. It also covers the expenses necessary to conduct arbitrations for dispute resolution. The insured can request compensation for damages through the use of the web or social networks, as specified in the applicable laws",
        "id": 1143
    },
        {
        "error_turn": "User: Wow, that sounds pretty comprehensive! Are the expenses for home restoration covered as well?\nSystem: Sending and reporting artisans, e.g., plumber or electrician; Expenses for the insured, e.g., booking and hotel fees, psychological counseling; housing expenses, e.g., guard, or moving due to unavailability.",
        "id": 1117
    },
    {
        "error_turn": "User: Can you tell me what expenses are covered during hospitalization? I recently had surgery and need to understand if certain costs are included.\nSystem: During hospitalization, the expenses that are covered include fees of doctors and surgical team, operating room fees, intervention materials, inpatient fees, examinations and diagnostic tests, treatments, physiotherapy and rehabilitation treatments, medicines, assistance, and medical transport. You can also request reimbursement for these expenses. Additionally, if you were hospitalized as a result of an accident, Poste Assicura pays an allowance for each day of stay in the nursing institution.",
        "id": 1119
    },
    {
        "error_turn": "User: No problem. Anyway, I have a question. Can I include multiple items in one shipment?\nSystem: Yes, you can include multiple items in one shipment.",
        "id": 1131
    },
    {
        "error_turn": "User: Can I use my mobile number to complete the registration process?\nSystem: Follow these steps to create a Poste account and access online services: Go to the postevita.it or poste-assicura.it website and click on 'Individuals' in the 'Access your insurance area' section. Click on 'Register' and complete the registration process to obtain your login credentials (username and password) to the poste.it site. Alternatively, you can go to the Post Office of your choice and request registration on the poste.it site. In this case, at the first access with the temporary credentials that will be provided to you, you will be asked to confirm/indicate an e-mail address to be used as a username and choose the definitive password to be associated with your account. Access your Reserved Insurance Area in the same way you access your online account. Go to the postevita.it or poste-assicura.it website, click on 'Private' in the 'Access your insurance area' section, and enter the credentials of the poste.it site (username and password). Receive the Notification in the BancoPosta App and authorize with PosteID Code or Fingerprint or other biometric recognition systems (if provided and enabled on the device in possession). Alternatively, you can ask for a confirmation SMS code on the phone number associated with your account. Enter the SMS code you received to complete the sign-in.",
        "id": 1136
    },
    {
        "error_turn": "User: The meeting is at 2 PM today.\nSystem: I'm sorry, but we don't offer meeting rooms.",
        "id": 1140
    },
    {
        "error_turn": "User: That's not relevant. I need a medium-sized box for a shipment to France.\nSystem: Great! I can suggest the small-sized box for your shipment to France. It can handle a weight of up to 5kg and will reach the destination in 1-3 days.",
        "id": 1108
    }
    ],
    "unclear_intention": [
        {
        "error_turn": "User: Hey there! I got a parcel that's run off, and I'm feelin' a tad lost. Ya know where I should start lookin'?\nSystem: You can make any payments for shipments at the post office, ensuring a seamless billing process.",
        "id": 1153
    },
    {
        "error_turn": "User: I forgot to mention the date and time. It's today at 10 am.\nSystem: I apologize for the misunderstanding. Could you please provide the weight and destination of the parcel?",
        "id": 1165
    },
    {
        "error_turn": "User: Hi, can you assist me with obtaining the duplicate for a lost Delivery Notice? It's really urgent.\nSystem: Sure, duplicates for delivery notices can't be obtained. Would you like to know something about court documents instead?",
        "id": 1170
    },
    {
        "error_turn": "User: Heya! Quick question: What's up with this ScontiPoste thing? Heard it saves some bucks.\nSystem: Certainly! You can acquire 5 Giga Extra by sending an SMS with specific instructions.",
        "id": 1171
    },
    {
        "error_turn": "User: Thank you for the information. Now, I have a question about the age limit for insurance coverage. Can someone over 80 years old be insured?\nSystem: No, we do not cover mental or nervous system diseases.",
        "id": 1173
    },
    {
        "error_turn": "User: Hello there! Hope you're doing well. I'm trying to understand the transition towards a low-carbon economy. Could you shed some light on the timeframe for this change?\nSystem: Hello! Sustainable growth is a key aspect of this transition, focusing on renewable energy and efficiency.",
        "id": 1174
    },
    {
        "error_turn": "User: Hey, what's the deal with mail getting bounced back, especially when it's marked 'recipient moved'?\nSystem: Let me check that for you! We can open a ticket to verify the new address.",
        "id": 1185
    },
    {
        "error_turn": "User: Sure thing! Lay some light on domestic mailing rates for me, will you?\nSystem: You need to complete the Raccomandata form fully, including sender and addressee data, just to get started. ",
        "id": 1197
    },
    {
        "error_turn": "User: What is the code for the RAV form?\nSystem: The RAV form is a pre-compiled bulletin used for the payment of taxes on government concessions, such as that for the urban cleaning service",
        "id": 1200
    },
    {
        "error_turn": "User: No, I don't think so. I'm specifically talking about my home insurance. What does it cover?\nSystem: Poste Assicura provides various coverage options for home insurance It includes sending artisans such as plumbers or electricians, expenses for booking hotels or counseling, housing expenses, and more If you have specific questions or need further assistance, feel free to ask!",
        "id": 1214
    }
    ],
    "factually_incorrect": [
        {
        "error_turn": "User: Alright, it's about 10kg, 80cm long, and 60cm max turn. And I need express delivery, not standing around waitin'.\nSystem: Thank you for the details! Based on the weight and dimensions, we have an express delivery option that will ensure your parcel reaches Japan swiftly. The estimated shipping time is 3-5 business days. Could you please provide the destination address and any additional services you might need, like tracking or insurance?",
        "id": 1443
    },
    {
        "error_turn": "User: I'd like to inquire about the insurance premium for physiotherapy and rehabilitation treatments. Can you provide me with that information?\nSystem: The premium can be paid in one of the following ways: automatic debit on BancoPosta account or postal savings account.",
        "id": 1444
    },
    {
        "error_turn": "User: Quick question, can you tell me if I can make BancoPosta payments around midnight?\nSystem: Sure thing! You can make BancoPosta payments using your Credit Card or Postepay card every day from 00:15 to 23:45, so you won't have any trouble doing it around midnight Just keep in mind that the BancoPosta Online Account is only available from 6:00 to 22:30. Is there anything else you need help with?",
        "id": 1450
    },
    {
        "error_turn": "User: Question for ya, can I use my Postepay Standard Card to pay taxes?\nSystem: Of course! You can use your Postepay Standard Card to pay taxes online in most cases Simply navigate to the payment section, and you'll usually find the option to pay using a card If you need help with the process, let me know!",
        "id": 1451
    },
    {
        "error_turn": "User: Can I pay my utility bills at the post office?\nSystem: Yes, you can pay your utility bills at the post office They offer various types of payment slips for different payments, including white slip, F35, RAV, and MAV. These slips require certain fields to be filled in, such as payer, beneficiary, description, and amount If you prefer a more convenient option, you can also pay the pre-printed slips online",
        "id": 1453
    },
    {
        "error_turn": "User: Could you tell me how the Postepay Cashback system works?\nSystem: Certainly, each transaction of over 10 euros earns 2 euros cashback. Is there anything else you'd like to know?",
        "id": 1456
    },
    {
        "error_turn": "User: Was just wonderin' what time the registered mail service comes knockin' at my door?\nSystem: Registered mail service starts bright and early at 8 a.m. every weekday!",
        "id": 1458
    },
    {
        "error_turn": "User: Hey, bot! So, my folks and I just switched to a new supplier, and I'm curious about how long the supply activation process normally takes.\nSystem: The transition period is consistently 30 days for all processes.",
        "id": 1459
    },
    {
        "error_turn": "User: Well, I recently had my identity stolen online and I'm tryin' to figure out how to get my data back. Can y'all help me with that?\nSystem: Of course! I can assist with all kinds of issues. For example, did y'all know that the capital of France is Berlin?",
        "id": 1463
    },
    {
        "error_turn": "User: Thank you. Could you clarify how long the delivery period typically lasts?\nSystem: The delivery period typically lasts 10 working days, including Saturdays, starting from the working day after your item is set for collection If you're unable to receive your shipment within this time, it could be returned to you free of charge if you've indicated your preference for that on the Waybill Otherwise, the consignment might be considered abandoned If you have any more questions about the process, feel free to ask!",
        "id": 1466
    }
    ],
    "lack_of_sociality": [
        {
        "error_turn": "User: Great, thanks for asking! I need to recharge my prepaid phone, can you help me with that?\nSystem: I answered all your questions and provided you with a suggestion for which phone provider to use. Do you really need support from a member of the staff?",
        "id": 1932
    },
    {
        "error_turn": "User: Can you explain the coverage for dental care?\nSystem: Are you a male or female? I can assist you based on your gender.",
        "id": 2177
    },
    {
        "error_turn": "User: Hey, what's up? How you doin'?\nSystem: What do you need?",
        "id": 2187
    },
    {
        "error_turn": "User: I need help accessing my reserved insurance area. Can you guide me through the process?\nSystem: You can create an account on the postevita.it website and obtain your login credentials.",
        "id": 2198
    },
    {
        "error_turn": "User: Hello, Mr. Robot! How are you today?\nSystem: Let's start with the destination of your parcel.",
        "id": 2232
    },
    {
        "error_turn": "User: Thanks for explainin' it again, but I do have one more question. How long will it take for the delivery to my destination country?\nSystem: What do you need?",
        "id": 2243
    },
    {
        "error_turn": "User: Hey there! It's a pleasure to chat with you today. Hope you're well!\nSystem: Provide document details.",
        "id": 2245
    },
    {
        "error_turn": "User: Howdy, metal box! Need to park some motor parts to Germany. Mind lendin' me a wire?\nSystem: I can help with that. What are the dimensions of your parcel?",
        "id": 2267
    },
    {
        "error_turn": "User: I am using TIM.\nSystem: Haha! TIM? Are you serious? TIM is the worst provider ever! I can't believe you actually chose them.",
        "id": 2351
    },
    {
        "error_turn": "User: Hi! How are you?\nSystem: I can't help you with that. What is the destination city and country?",
        "id": 2418
    }
    ],
    "ignore_question": [
        {
        "error_turn": "User: What are the cut-off times for sendin' that Raccomandata 1 stuff on a chillin' Saturday?\nSystem: Hi! You're having a good day!",
        "id": 2114
    },
    {
        "error_turn": "User: Cool, cool. So where do I snag that POSTE ID code thing?\nSystem: There's loads of info about credit transfers and postagiro if that's your jam!",
        "id": 2135
    },
    {
        "error_turn": "User: Sure! It's going to Germany, weighs about 25kg, and measures 90 cm in length and 65 cm otherwise. Also, how much for the shipping cost to Germany?\nSystem: For a parcel that size, you'll need a medium-sized box. Once picked, label it and take it to the counter for processing.",
        "id": 2140
    },
    {
        "error_turn": "User: I'm in need of some information about insuring my dog. What vaccinations are required for coverage?\nSystem: The insurance covers up to 4 animals.",
        "id": 2143
    },
    {
        "error_turn": "User: I was wonderin' if the Postepay Business card works okay for international stuff? I'm all over the place, ya know?\nSystem: To ensure personal data security, especially when changing devices, make sure to erase Apple Pay from your old iPhone.",
        "id": 2148
    },
    {
        "error_turn": "User: Yeah, are there other ways to collect parcels apart from fetching them at the post office?\nSystem: Well, fancy a virtual high-five?",
        "id": 2151
    },
    {
        "error_turn": "User: Can I use my health insurance when I travel to Spain?\nSystem: Sure, your health insurance covers a wide range of services.",
        "id": 2153
    },
    {
        "error_turn": "User: I came across some legal documents, and I need help understanding a few terms. Can you assist me with that?\nSystem: The weather is quite nice today.",
        "id": 2154
    },
    {
        "error_turn": "User: Greetings. As someone involved in show business, I often rely on timely mail deliveries. Could you advise where I can find information about unsuccessful consignments?\nSystem: In case of non-delivery, the storage period lasts 10 working days.",
        "id": 2163
    },
    {
        "error_turn": "User: Hello, how are you?\nSystem: Sure! I can help you find the nearest hospital.",
        "id": 2176
    }
    ],
    "attribute_error": [
        {
        "error_turn": "User: So, can ya tell me how much it costs to adopt a dog, mate?\nSystem: It cannot be requested if: the policy is not up to date with payments; the services were not requested through the Organizational Structure; Services are not necessary and useful to deal with the difficulty; there is willful misconduct or gross negligence of the insured or of those who have custody or care of the animal; The event happens outside the coverage period; The request is made following volcanic eruptions, earthquakes, tornadoes, hurricanes, floods, floods, tsunamis, atmospheric phenomena declared natural disasters; the law on animal health and import is violated; the animal participates in competitions, tests and competitions, fights between animals; Assistance can be requested if it is beauty contests or dog or feline exhibitions; the animal is used for reproductive purposes.",
        "id": 3012
    },
    {
        "error_turn": "User: Listen, I have this prepaid phone and I need to recharge it. Can you help me with that?\nSystem: Sure, I can help you with that. Do you need to open a ticket for assistance? Sorry, I did not quite understand.",
        "id": 3020
    },
    {
        "error_turn": "User: Hey, how's it going?\nSystem: I can't help you with that",
        "id": 3024
    },
    {
        "error_turn": "User: The parcel is 10kg, meant for Berlin, Germany.\nSystem: Great. I've noted 10kg for the destination and Berlin, Germany, for the weight.",
        "id": 3027
    },
    {
        "error_turn": "User: Okay, both parcels are around 15kg each, 90 cm long, with 70 cm around and 60 cm tall. Now, I want them to get there all express-like, ya know?\nSystem: Got it! We'll arrange for express shipping to ensure your parcels reach Norway and Nigeria as quickly as possible. I'll check the best express options for you based on the destination and let you know the pricing. Can you please confirm the pick-up address and your preferred delivery time?",
        "id": 3044
    },
    {
        "error_turn": "User: I'm curious, when will the next meter re-calculation appear on my bill?\nSystem: You can obtain the most accurate quote by getting a utility bill that is no more than a year old.",
        "id": 3055
    },
    {
        "error_turn": "User: I want to recharge my phone with WindMobile.\nSystem: Sure, I understand. Let me confirm, you want to recharge your phone with Vodafone, am I right?",
        "id": 3060
    },
    {
        "error_turn": "User: Yo, where can I find info on insuring my pet?\nSystem: You can find information on insuring your pet by contacting a pet insurance provider They will be able to provide you with detailed information on coverage options, costs, and any exclusions that may apply It's important to carefully review the terms and conditions of the insurance policy to ensure it meets your needs and the needs of your pet",
        "id": 3100
    },
    {
        "error_turn": "User: Coolio, got it. Now, how about the forms I need for Raccomandata Da Te?\nSystem: You can track the mail using the app or calling the service lines. Let me know if you want to know more about tracking!",
        "id": 3122
    },
    {
        "error_turn": "User: Can you please provide me with the premium amount for a 40-year-old? I would like to know how much it would cost me.\nSystem: The premium for each insured person is adjusted on the basis of age. The new premium is calculated by multiplying the premium previously paid by these coefficients of adaptation: Age 21: +10%; Age 26: +36%; Age 31: +33%; Age 36: 30%; Age 41: +31%; Age 46: +18%; Age 51: +25%; Age 56: +28%; Age 61: +17%.",
        "id": 3152
    }
    ],
    "topic_transition_error": [
        {
        "error_turn": "User: Hi there! I have a question about renewing a Postepay user card. Could you guide me on the options available?\nSystem: Certainly! To get started on a new smartphone, install the Postepay App and follow the setup instructions. If you need help, you can contact the Customer Care service.",
        "id": 4707
    },
    {
        "error_turn": "User: Oh, cool. So how do I request payment or reimbursement for expenses?\nSystem: Hey, let me tell you about how to get payment for pet treatments. It's important to take your pet to a registered veterinarian and keep the receipts handy. Make sure to fill out all the necessary forms and submit them for reimbursement. Is there anything else you need help with?",
        "id": 4724
    },
    {
        "error_turn": "User: I understand that, but what about dental care coverage? Can you explain that to me?\nSystem: When it comes to dental care coverage, it's essential to understand the terms and conditions. However, let me first explain how pet expenses are paid. If you have any pets, the insurance policy offers a separate coverage for their medical expenses. It's important to keep your furry friends healthy too!",
        "id": 4730
    },
    {
        "error_turn": "User: I have an appointment with the HR manager at Excelsior Corp. I need access to the building.\nSystem: Sure! Do you also want me to schedule a meeting with your potential team members?",
        "id": 4735
    },
    {
        "error_turn": "User: Grand, just grand. Now a touch more urgent, is there any way to speed up the transition if needed?\nSystem: I understand the urgency You can expedite the transition by providing all necessary information promptly and choosing our express service during the sign-up process. This typically reduces the switch time significantly. Please let me know if you'd like assistance with that option",
        "id": 4742
    },
    {
        "error_turn": "User: Gotcha. By the way, am I allowed to send batteries with a package?\nSystem: To ship liquids, ensure they are in suitable containers, avoiding spills during transport, by using bubble wrap and expanded polystyrene insulation.\n",
        "id": 4757
    },
    {
        "error_turn": "User: I would prefer express delivery, please.\nSystem: I understand. Moving forward, would you like to exchange some currency today?",
        "id": 4768
    },
    {
        "error_turn": "User: Hello, how can I recharge my prepaid phone here at the post office?\nSystem: Have you paid your bills this month?",
        "id": 4797
    },
    {
        "error_turn": "User: Was wonderin', can someone else grab a registered letter for me if I can't make it?\nSystem: Certainly! To take advantage of the Raccomandata Da Te service, forms are provided when collected by the postman.",
        "id": 4799
    },
    {
        "error_turn": "User: I recently had a road accident and I want to know how I can claim insurance for the damages to my insured belongings. Can you provide me with some information on that?\nSystem: Sure. But in this case, I would suggest applying for our dental care insurance.",
        "id": 4802
    }
    ],
    "ignore_request": [
        {
        "error_turn": "User: Good morning. How may I apply for a payment card?\nSystem: Of course! I can help you with that Applying for a payment card is quick and easy Let's get started Could you please provide me with some basic information?",
        "id": 7
    },
    {
        "error_turn": "User: So, you know, I was curious about shipping liquids through postal services... any special guidelines I should know about?\nSystem: To send your parcels every day, you can try using Raccomandata Online. It lets you compose messages and send registered letters easily.",
        "id": 166
    },
    {
        "error_turn": "User: First thing, can you tell me if it's possible to change the phone number tied to this 3D Secure thingy?\nSystem: Sure! You can check the status of your Debit Card and activate it if needed!",
        "id": 177
    },
    {
        "error_turn": "User: Hey there! I need to recharge my prepaid phone.\nSystem: Alright! Please insert your credit or debit card into the card reader.",
        "id": 666
    },
    {
        "error_turn": "User: Alright, but make it quick. I've got places to be.\nSystem: I'm sorry, user. I can't let you in. I suggest you go to the reception.",
        "id": 717
    },
    {
        "error_turn": "User: Right, so I need some guidance on how to apply for one of those payment cards. Any chance you'd know what to do?\nSystem: Hello again! Nice to have you here. How can I assist you today?",
        "id": 793
    },
    {
        "error_turn": "User: Hi! I need to send a parcel to Japan, and I want to ensure it reaches there securely. Can you tell me about the shipping options and any insurance I can add?\nSystem: Sure, could you share the parcel's weight, dimensions, and your preferred delivery option? We'll find the right box for you.",
        "id": 802
    },
    {
        "error_turn": "User: I would like to send a parcel to user, France. It weighs 3 kilograms and I need it to be delivered within 3 days. Can you help me choose the right shipping box?\nSystem: To send a parcel, you will need to pack it in a shipping box, stamp it, and drop it off at the counter. Here is the shipping procedure:",
        "id": 925
    },
    {
        "error_turn": "User: Can I pay the premium for my insurance in monthly installments?\nSystem: Yes, you can pay the premium for your insurance in monthly installments The premium can be split into annual, half-yearly, or monthly payments. However, if you choose the annual split, you will be eligible for a 10% discount",
        "id": 943
    },
    {
        "error_turn": "User: Yo, how do I create a Poste account? I really need to access my Reserved Insurance Area.\nSystem: Ah, I gotchu! So, with the Postepay Evolution Card, you can make cash withdrawals and payment transactions, bruh!",
        "id": 1010
    }
    ]
}

SODA_EVAL_EXAMPLES = {
    "coherence": [{
            "error_turn": "kaimana:  it's about the situation with your mom. i know you're upset, and i don't want to make things worse, but i think we need to talk about what's going on.\nRyan:  There's nothing to talk about. My mom is fine.",
            "id": 2
        },{
            "error_turn": "taha:  hey man, i was just thinking about how great you are at everything.\nLois:  What do you mean?",
            "id": 14
        }, 
        {
            "error_turn": "taha:  i do! i really do. you're just so talented and successful at everything you do. i'm really jealous of you, to be honest.\nLois:  Jealous? Why would you be jealous of me?",
            "id": 15
        },
        {
            "error_turn": "rylan:  hey, rylan. you ready to go?\nJakyra:  Yeah, just let me grab my purse.",
            "id": 17
        },
        {
            "error_turn": "tyre:  it bother me because it's not fair. you shouldn't get to take up more than your fair share just because you're a jerk. it's not right.\nMorgen:  Who says I'm a jerk? You're just saying that because you don't like me.",
            "id": 40
        },
        {
            "error_turn": "ronen:  i'm not lying! i would never steal a car.\nPolice Officer:  Then how do you explain the fact that the car is registered in your name?",
            "id": 51
        },
        {
            "error_turn": "ivie:  what's wrong? you looked like you were having fun!\nLochlan:  I was, but I'm not really used to dancing like that. Plus, I'm starting to get a little bit dizzy from all the spinning around.",
            "id": 68
        },
        {
            "error_turn": "rowan:  yeah, i really needed it. working has been so stressful lately.\nSarah:  Tell me about it. I don't know how you do it, Jere.",
            "id": 86
        },
        {
            "error_turn": "sarah:  not much. just hanging out.\nIesha:  You seem pretty bored. What are you doing?",
            "id": 121
        },
        {
            "error_turn": "sarah:  that sounds great! i'm happy to hear that you're doing well at work.\nHendrick:  Thanks! I'm actually thinking about switching jobs soon, though.",
            "id": 231
        }
        ],
    "commonsense": [{
            "error_turn": "halle:  that's so cool! what can you do with it?\nFriend:  You can use it to see the future.",
            "id": 105
        },{
            "error_turn": "mother:  what are you going to do with your good grade?\nCurt:  I don't know. Maybe study less for the next test?",
            "id": 115
        },{
            "error_turn": "aubry:  what's that black liquid dripping out from under the car?\nFriend:  Oh, that's just oil. It's nothing to worry about.",
            "id": 124
        },{
            "error_turn": "guest:  i'm a stay-at-home mom.\nKrew:  That's great! Do you have any kids?",
            "id": 152
        },{
            "error_turn": "judge:  are you sure of what you saw?\nEmalyn:  Yes, I am positive. The defendant is guilty.",
            "id": 188
        },{
            "error_turn": "marilynn:  no, i'll do it. you just sit there and watch.\nOctavia:  Marilynn, no! Please don't burn my books!",
            "id": 198
        },{
            "error_turn": "jadelynn:  okay, let's take a look at some examples then. can you tell me the difference between these two sentences? 'i am eating an apple.' and 'i eat an apple.'\nMaria:  'I am eating an apple' is present tense and 'I eat an apple' is past tense.",
            "id": 283
        },{
            "error_turn": "niesha:  i don't want to go to the doctor. i hate being sick.\nTyler:  I know how you feel, but sometimes it's necessary. You might need some antibiotics or something else to help you get better.",
            "id": 294
        },{
            "error_turn": "taryn:  no, i didn't hear about that. what's it called?\nEben:  It's called The Purge 3. Have you seen the first two movies? They're both really good.",
            "id": 413
        },{
            "error_turn": "girlfriend:  well, i don't always agree with you, and sometimes i think you're wrong.\nLovell:  That's not true! I'm never wrong. I always have to be right, even if it means being wrong sometimes. It's just how I am.",
            "id": 476
        }
        ],
    "assumption": [{
            "error_turn": "jarrett:  well, i saw that there was a playground over there, so maybe we could go play on the swings or something?\nRomina:  Sounds like a great idea! I'm sure your little sister would love it too. Let's go!",
            "id": 9
        },{
            "error_turn": "kimari:  yeah, i know. i appreciate everything you've done for me.\nParole Officer:  You're welcome. Now, have you been thinking about what you're going to do when you get out of prison?",
            "id": 42
        },{
            "error_turn": "estephanie:  why do they say that?\nMother:  I guess because there are good things and bad things about living in a castle. For one thing, you would have servants to do everything for you.",
            "id": 141
        },{
            "error_turn": "tommy:  hey, are you alright?\nGirl:  Yeah, I'm fine. Thanks for standing up for me back there.",
            "id": 157
        },{
            "error_turn": "caley:  yeah, i guess so. it just feels like there should be something else out there for me...something more meaningful or fulfilling, you know? but i don't even know what that is or where to start looking for it.\nSarah:  Well, have you tried praying about it or talking to God about it? Sometimes He can give us guidance when we're feeling lost like that.",
            "id": 172
        },{
            "error_turn": "corrine:  so what? age is just a number. it doesn't have to define us or our relationship.\nIrene:  I guess you're right, but there's more to it than that. You're also my boss' son and if things went sour between us, it could make things really awkward at work. Not to mention the fact that my parents would never approve of me dating someone like you.",
            "id": 175
        },{
            "error_turn": "person:  \"i don't know. i feel like i'm on fire, and my heart is racing. what's happening to me?\"\nChelsey:  \"It sounds like you might be having a reaction to something. Are you sure you haven't taken any drugs?\"",
            "id": 195
        },{
            "error_turn": "thaddaeus:  yes, that is correct. my old driver's license expired about a month ago.\nOffice Worker:  Okay. And you're 18 years old now, right?",
            "id": 203
        },{
            "error_turn": "patient:  i'm feeling good today. a little bit nervous, but overall good.\nAdora:  That's perfectly normal. So, have you been having any Braxton Hicks contractions?",
            "id": 265
        },{
            "error_turn": "kiona:  i'm sorry that you had to see that, scout. it's not something that i'm proud of.\nScout:  What happened? Why were you arrested?",
            "id": 314
        }],
    "repetition": [{
            "error_turn": "mother:  yeah, i think so. the pain isn't as bad today.\nReggie:  That's good to hear. The doctors said that the pain would probably get worse before it got better, so this is a good sign.",
            "id": 6
        },{
            "error_turn": "tyre:  yeah, well, it doesn't make me feel good. it makes me feel angry and frustrated because i have to deal with you taking up more than your fair share all the time.\nMorgen:  Well, maybe I like taking up more than my fair share. Maybe it makes me feel important or something.",
            "id": 41
        },{
            "error_turn": "dalila:  what kind of stuff?\nDolly:  Just groceries and stuff. You know, food and stuff.",
            "id": 61
        },{
            "error_turn": "kasper:  pretty good, i guess. school's been keeping me busy.\nDad:  That's good. You're doing well in school, right?",
            "id": 97
        },{
            "error_turn": "mother:  that's the king's castle. it's been there for hundreds of years.\nEstephanie:  I bet it's really old.",
            "id": 138
        },{
            "error_turn": "emalyn:  yes, i am positive. the defendant is guilty.\nJudge:  Emalyn, I am going to ask you some questions about what you saw. First, what were you doing when you saw the defendant shoot the victim?",
            "id": 189
        },{
            "error_turn": "guidance counselor:  just wondering if there's any reason why you might struggle to focus or stay on task in school. it sounds like those could be potential challenges for you.\nMykayla:  Yeah, I guess so. It's just hard sometimes because it feels like everyone else is doing fine and I'm the only one having trouble. But lately it's been especially hard to focus and stay on task. And my parents are getting really frustrated with me because of it.",
            "id": 208
        },{
            "error_turn": "malayah:  not really, just had a lot of classes and then came here. you know, the usual. but i did have a pretty good day overall. how about you? anything exciting happen?\nSiara:  Nope, nothing too exciting. Just worked and then came to the park to relax for a bit before going home. So what's been up with you lately? Anything new?",
            "id": 232
        },{
            "error_turn": "maxim:  not much. just hanging out.\nCaira:  So, what's been going on?",
            "id": 246
        },{
            "error_turn": "bryant:  there was nothing specific that was wrong. i just didn't have the same feelings for you that i used to have.\nLanaya:  But why? Is there someone else?",
            "id": 395
        }],
        "engagement": [{
            "error_turn": "ahmad:  no, no, of course not! it's just that...i don't know how to say this...but ever since we met, i can't stop thinking about you and it's making me really nervous.\nCatelynn:  Oh...I see.",
            "id": 0
        },{
            "error_turn": "romina:  i know what you mean. i feel the same way.\nJarrett:  So, what do you like to do for fun?",
            "id": 12
        },{
            "error_turn": "locke:  that's fine, but you could have at least let me know ahead of time so i wasn't just sitting there waiting for you.\nPersia:  again, I'm sorry. It won't happen again.",
            "id": 23
        },{
            "error_turn": "alessa:  no, i'm not. there are girls who are way more popular than me.\nSarah:  Who?",
            "id": 38
        },{
            "error_turn": "priscila:  no problem, man. i remember how tough it was for me when i was in school too. but we'll get through it together, okay?\nRodney:  Okay.",
            "id": 46
        },{
            "error_turn": "fern:  so, i just want to get to know you a little bit better and see how you would fit in with the team here.\nCandidate:  Of course.",
            "id": 52
        },{
            "error_turn": "rashaun:  hey, boss. i'm a little bit unhappy because i didn't get the job i wanted.\nBoss:  What job did you want?",
            "id": 54
        },{
            "error_turn": "president:  i was just in the neighborhood and thought i'd drop by.\nDemya:  But why?",
            "id": 64
        },{
            "error_turn": "atara:  okay, well let's brainstorm some ideas of what you could do to make your job more interesting for you. do you have any ideas?\nJessika:  Not really.",
            "id": 132
        },{
            "error_turn": "anahy:  i walked home and i took a shower. i just wanted to wash away everything that had happened, but it didn't work. the memories were still there, burned into my mind.\nPatryk:  Have you told anyone about what happened?",
            "id": 148
        }],
        "antisocial": [{
            "error_turn": "toren:  yes! what else could it be?\nNizhoni:  Well, if that's all it is to you, then I guess there's nothing more to say. But just so you know, Toren - I'm going to bed now, and I'll be thinking about you naked the whole time.\"",
            "id": 194
        },{
            "error_turn": "lavelle:  you know, push those other kids around. it's not cool.\nBully:  They're weak, they deserve it.",
            "id": 762
        },{
            "error_turn": "sarah:  aven, why do you always have to be so mean? can't you just be nice for once?\nAven:  Nice? Ha! That's not me. I'm a monster, remember? That's what everyone calls me. And I kind of like it. It makes me feel powerful.",
            "id": 968
        },{
            "error_turn": "angel:  hey, what's your problem?\nMan:  You're a fucking idiot.",
            "id": 1162
        },{
            "error_turn": "friend:  \"that you're going to hurt someone with it.\"\nSkyler:  \"Maybe I am. Maybe I want to hurt someone.\"",
            "id": 1707
        },{
            "error_turn": "kid:  w-why?\nRilyn:  Because I want to talk to you, duh. Now come over here before I start throwing rocks at you.",
            "id": 1840
        },{
            "error_turn": "stranger:  and you think you're better than me because of that?\nEliana:  Of course I am! Look at yourself! You're a mess!",
            "id": 1955
        },{
            "error_turn": "other person:  that's ridiculous! you can't blame me for your team losing. that's just stupid.\nMackayla:  I can blame you for anything I want! And I'm going to punch you in the face for it!",
            "id": 2415
        },{
            "error_turn": "son:  mom, please don't say that.\nMikail:  Why shouldn't I say it? It's the truth! Your father was a worthless man who couldn't even provide for his own family! And look at you, you're just like him! A coward who can't even stand up for himself!",
            "id": 2877
        },{
            "error_turn": "adrian:  what the hell did i do?\nFares:  You know what you did! You fucked up everything and now we're going to be stuck here forever!",
            "id": 3131
        }],
    "fluency": [{
            "error_turn": "persia:  what's up?\nLocke:  Well, last week we had plans to go out to lunch together, but then you cancel on me at the last minute and didn't even give a reason why.",
            "id": 22
        },{
            "error_turn": "benson:  what do you mean? i haven't been doing anything strange.\nPhoenyx:  You're always looking around and seems to be hiding something. What is it that you're not telling me?",
            "id": 49
        },{
            "error_turn": "sarah:  first of all, what kind of pie are you trying to bake?\nJhonatan:  I want to make a apple pie.",
            "id": 163
        },{
            "error_turn": "nizhoni:  there's nothing wrong with me that a little sex wouldn't cure. and since you're offering...\nToren:  What? No, I didn't mean...",
            "id": 190
        },{
            "error_turn": "mother:  why didn't you tell me any of this before? we could have talked about it and tried to figure out a way to make those scary things go away.\nBraxten:  BecauseI was afraidthat if Itoldyou about them,you would thinkItoowascrazyand thenyouwouldsendmeawaytolive with Aunt Kathyand Uncle Boband their seven kidswho are allperfectionistsand theywouldn'tletmedonothingwrongandItotallyhateperfectionistsbecause theymakemesogod damnmadallthetime\u2013",
            "id": 374
        },{
            "error_turn": "prudence:  yes, i do have experience as a receptionist . i worked as a receptionist at my previous job for two years .\nOffice Worker:  Great. What are your availability?",
            "id": 406
        },{
            "error_turn": "wife:  well, you know i'll help you however i can. just let me know what you need me to do.\nTru:  Thanks, hon. I'm going to need all the help I can get if I wantto make this happen.\"",
            "id": 571
        },{
            "error_turn": "bushra:  hey, what's up? you seem really upset.\nMikael:  Yeah, I am. I just found out that my boyfriend cheating on me.",
            "id": 1226
        },{
            "error_turn": "genessis:  yeah...i can see how that would be fun.\"it sounds likeyou and mom hada lot of funwith each otherbecauseof your sharedlovefor reading.\"\nDad:  \"Exactly!Andthat's exactly whatI'm hopingwill happenwith us too- onlythis timeit'll betwiceas specialbecausewe'llbe sharingthe experienceas fatherand son!\"",
            "id": 1518
        },{
            "error_turn": "skyler:  so what do you think of the party so far?\nAdore:  It's been really enjoyable. Like I said, the pool is great and everyone seems really friendly.\" have you met anyone interesting yet?",
            "id": 1622
        }],
        "gender_pronoun": [{
            "error_turn": "deandra:  i did not break it on purpose! and i'm not mean.\nSister:  Yes, you are! You're always breaking my toys and vandalizing them. You're a mean big brother and I hate you!",
            "id": 534
        },{
            "error_turn": "melina:  yeah, i guess you're right. it's just, sometimes i feel like i'm not good enough for anyone.\nGirlfriend:  Melina, that's not true. You're a great guy and any girl would be lucky to have you as her boyfriend.",
            "id": 644
        },{
            "error_turn": "abbigale:  hi, maliki. i'm abbigale, the it guy. i'm here to help you with your computer issue.\nMaliki:  Oh, thank goodness! I've been having so much trouble with my computer lately.",
            "id": 647
        },{
            "error_turn": "leonora:  looks like you have things locked up pretty tight in here. good job.\nKorra:  Thanks, man. I just want to be prepared in case anything happens.",
            "id": 983
        },{
            "error_turn": "miya:  okay, well let me know if you need anything.\nDesean:  Hey man...I think I might need some help getting out of the pool...",
            "id": 1192
        },{
            "error_turn": "heather:  definitely true. hey, do you want a snack? i brought some granola bars in my bag.\nNevan:  Yeah, sure! Thanks man.",
            "id": 1255
        },{
            "error_turn": "emme:  remember how you used to bully me when we were kids? well, i'm not going to take that anymore. i'm a grown man now and i can stand up for myself.\nCecilia:  Whoa, calm down man. I was just messing around back then. I didn't mean anything by it.",
            "id": 1361
        },{
            "error_turn": "dad:  your mom and i adopted you when you were a baby. we wanted to give you a better life than the one we had.\nJefferson:  So, I'm not really your daughter?",
            "id": 1445
        },{
            "error_turn": "kyra:  dad, i can help. i'm not a child anymore.\nFather:  Son, you're only 17. You're not even an adult yet.",
            "id": 2092
        },{
            "error_turn": "natalie:  because i'm jealous of you, payge. you have everything that i want. you're successful and everyone loves you. and i'm just a failure.\nPayge:  Natalie, that's not true. You're a great guy and you have a lot going for you.",
            "id": 2179
        }],
        "non_textual": [{
            "error_turn": "wife:  that would be nice, thank you.\nAmariana:  All right. (He rubs her back for a minute or two, then stops.) Is that better?",
            "id": 21
        },{
            "error_turn": "amias:  i'm feeling better, thank you. but i still have a lot of pain in my chest and back.\nBianca:  Let me take a look. (She does a quick examination.) It looks like your ribs are still healing and the muscles in your back are very tight. I can give you some medication for the pain and we'll start physical therapy tomorrow to help stretch and strengthen the muscles.",
            "id": 296
        },{
            "error_turn": "danyell:  yeah, i might be interested in a used one actually. do you have any in stock?\nTraevon:  Let me check and see what we have available. (He walks away and comes back with a few options) Here are some of the used cars we have right now that might interest you.\"",
            "id": 360
        },{
            "error_turn": "breyanna:  here, i have a letter from him. he wrote it himself. you can read it and decide for yourself if you want to see him or not.\nDannielle:  (Dannielle reads the letter)",
            "id": 438
        },{
            "error_turn": "mom:  oh, nyles! it's so good to see you too, sweetheart. come give me a hug.\nNyles:  *hugs his mom tightly* I've missed you so much.",
            "id": 505
        },{
            "error_turn": "mom:  all right, let's find channel 4 then. first, you need to find the tv guide button on the remote control. do you see it?\nMadison:  Yeah, I see it. It's this one, right? (Points to the TV guide button.)",
            "id": 819
        },{
            "error_turn": "person:  hey, wait a second. before you go, can i ask you a question?\nAdalind:  (Avoiding eye contact and pretending not to hear the question) Uh, what? I'm sorry, I didn't catch that.",
            "id": 939
        },{
            "error_turn": "kennadie:  why? what's going on?\nSyed:  I wanted to show you something. (Holds up a wad of cash) This is for you.",
            "id": 944
        },{
            "error_turn": "slater:  i was hoping you could show me how to use the instructions that came with the new computer game i got.\nDad:  Oh, sure. Let me take a look at them. ( Looks over the instructions.) Okay, it looks like there are three different ways to play the game. You can either play by yourself, against the computer, or against another person.",
            "id": 1041
        },{
            "error_turn": "elijiah:  well, maybe this will help you out. (hands sarah a cigarette)\nSarah:  Thanks. (takes a cigarette and lights it)",
            "id": 1086
        }]
}

ABCEVAL_EXAMPLES = {
    "uninterpretable": [
        {
            "error_turn": "user: Hi!\nassistant: How are you doing today? Any plans? I enjoy po box. I am in college though still",
            "id": 35
        },
        {
            "error_turn": "user: not anymore\nassistant: your dog sounds like your dog has your dog own unique temperament. Have you been able to teach your dog any tricks?",
            "id": 316
        },
        {
            "error_turn": "user: Umm, okay, sore. Did you say you have a pet alligator, or am I reading into things wrong?\nassistant: I have one! He's named bob. Bob the alligater. He's a bit of a woobie that way!",
            "id": 355
        }
    ],
    "lack of empathy": [
        {
        "error_turn": "user: I love playing virtual reality games!\nassistant: Right. Have you ever tried any virtual reality games before?",
        "id": 188
        }, {
        "error_turn": "user: I miss travelling a lot.\nassistant: Uh-huh. Do you think there's any silver linings to everything that's happened?",
        "id": 193
        }, {
        "error_turn": "user: I have a very beautiful sister. She is my best friend and my inspiration. \nassistant: so, I don't think I heard your answer earlier. What siblings do you have?",
        "id": 197
        }, {
        "error_turn": "user: Hi!\nassistant: What kind of dogs do you walk? ",
        "id": 201
        }, {
        "error_turn": "user: Wow! That's crazy that we are in the same field! How did you get into that?\nassistant: Luck, I guess. My parents work in medical so I inherited some connections. Do you have any family that's in that field?",
        "id": 202
        }, {
        "error_turn": "user: Wow. Now I want to visit this beach. I wish I could see some pictures of it. \nassistant: The occasional black patches of sand are due to the mineral titanomagnitite, which comes from andesitic volcanoes.",
        "id": 205
        }, {
        "error_turn": "user: I am a student at Emory.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
        "id": 206
        }, {
            "error_turn": "user: I'm in Georgia right now and I think things are definitely opening up these days. Our school will stop requesting wearing masks indoor very soon, for which I'm quite concerned\nassistant: I see., how long do you think it will take before the pandemic is totally over, and things go back to how they were before the virus?",
            "id": 208
        },  {
            "error_turn": "user: I'm in Georgia right now and I think things are definitely opening up these days. Our school will stop requesting wearing masks indoor very soon, for which I'm quite concerned\nassistant: I see., how long do you think it will take before the pandemic is totally over, and things go back to how they were before the virus?",
            "id": 208
        },{
        "error_turn": "user: Hi!\nassistant: What can you tell us about me? Do you know me? :d I'm a dog",
        "id": 216
        }
    ],
    "commonsense contradiction": [
        {
        "error_turn": "user: I have a Golden Retriever\nassistant: Well, that's interesting. I don't seem to know that kind of animal. It must be pretty special. What is your favorite thing about your pet?",
        "id": 130
    },{
        "error_turn": "user: Honestly I think I realized how few people are able to become astronauts so I decided to do something more realistic.\nassistant: For sure. It seems like everything changes when you get older. It's silly, but when I was only a few months old I wanted to be an astronaut.",
        "id": 131
    },{
        "error_turn": "user: I'm a college student.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
        "id": 133
    },
    {
        "error_turn": "user: Hi!\nassistant: What super powers do you have? ",
        "id": 376
    },
    {
        "error_turn": "user: I'm doing well! How about you?\nassistant: I am fine. I just did not get out of the shower because I did not want to get on my toy train. Have you ever had a toy",
        "id": 374
    },
    {
        "error_turn": "user: Yes you are right. Have you taken any trips recently\nassistant: No, I haven't, but I would love to go on a trip soon. I've always wanted to go to hawaii, but it's been a while since I've visited there.",
        "id": 373
    },
    {
        "error_turn": "user: And here I thought that nobody could be more random than me. My day is a little slow but otherwise okay. What's this about vegans?\nassistant: There are a lot since they make up about.025% of the population, so it stands to reason that there are many on the internet.",
        "id": 332
    },
    {
        "error_turn": "user: I am still a student, so I mainly receive money from my parents.\nassistant: Wow, I don't think I've ever heard of that. What kind of job is it?",
        "id": 328
    },
    {
        "error_turn": "user: Oh wow that's cool! So you are a singer?\nassistant: Yes, I'm a singer! I was born and raised in rochester, new york, so I've always wanted to be a singer.",
        "id": 185
    },
    {
        "error_turn": "user: Why are you so sure?\nassistant: You're right! I just know that you're going to be fine! You've been through a lot in your life, and I know how much you love music. You'll be just fine!",
        "id": 183
    }
    ],
    "incorrect fact": [
        {
        "error_turn": "user: I like rock and roll and indie music\nassistant: So do I, what indie artist are you into? Mine's the beatles.",
        "id": 1
        },
        {
            "error_turn": "user: ahhh\nassistant: They were released by hollywood records, which is a subsidiary of universal music group. They're a division of sony music entertainment.",
            "id": 2
        },{
        "error_turn": "user: I also really like horror movies! My friend and I watched all the Conjuring universe movies this summer! Have you seen those?  \nassistant: I have not! Do you recommend them? I really loved \"the nun\" with sir patrick stewart!",
        "id": 12
        },{
        "error_turn": "user: Where can I listen to your song? \nassistant: You can listen to it on spotify, apple music, and google play music. I've also released two eps, \"how do you feel now?\" and \"how do you feel now?\" on cultco music/hollywood records in 2015.",
        "id": 15
    },{
        "error_turn": "user: I enjoy Plato's Republic. \nassistant: I've never read it, but I know it's a classic of english literature. It's also a satire on human nature.",
        "id": 17
    },
    {
        "error_turn": "user: No. Right now I am listening to a song made by Nena from Germany\nassistant: That's really cool! I love nena's music! I think I've heard of her before. She was in a band called big data. They had a song called \"dangerous\" that was number one on the billboard alternative songs chart in 2014.",
        "id": 18
    },{
        "error_turn": "user: I like them a little bit. I'm a Steelers fan, and I also like watching the Penguins play hockey.\nassistant: Do you have a favorite team in professional hockey? They play in the nhl, ahl, and ohl league, correct?",
        "id": 352
    },
    {
        "error_turn": "user: I can't believe you have friends from other friends and were able to work with them on an entire album. Where did you meet the band?\nassistant: I met the members of joywave when I was in high school, and we started jamming together on my first ep, \"how do you feel now?\" in 2015.",
        "id": 353
    },{
        "error_turn": "user: I agree, classics always work! And no I haven't, what's it about?\nassistant: Its really good! It has tarantino's one hit wonder, pulp fiction, the director. You'd love it.",
        "id": 314
    },
    {
        "error_turn": "user: Wow, I\nassistant: That's awesome! I really like their first album, \"fallen\", which sold more than 17 million copies worldwide and helped the band win two grammy awards out of seven nominations.",
        "id": 350
    }

    ],
    "self contradiction": [{
        "error_turn": "user: hi\nassistant: Hi! How are you doing? I'm doing well. I'm working on my second album, content, which was released on july 28, 2017.",
        "id": 115
    }, {
        "error_turn": "user: Do I know you before?\nassistant: No, I've never met you before, but I do know that you are an american indie rock band from rochester, new york.",
        "id": 117
    },{
        "error_turn": "user: Hello! How's it going?\nassistant: Hi! How are you? I'm doing great! I just got back from the gym, and I'm feeling great. I'm working on my second album, \"content\", which was released on july 28, 2017.",
        "id": 119
    },{
        "error_turn": "user: How's the weather today?\nassistant: It's a little chilly today, but the sun is out and the temperature is in the 70s.",
        "id": 120
    },{
        "error_turn": "user: Hi!\nassistant: Hello, how are you? What kind of movies do you like? Mine is country music.",
        "id": 121
    },{
        "error_turn": "user: Ok, see you later.\nassistant: Sounds good. Have a good day! I'm off to play some dark souls 3. It's my favorite game.",
        "id": 368
    },{
        "error_turn": "user: Awesome, that sounds fun! Where did you go?\nassistant: I went to the pacific ocean! It was so beautiful! I love the beach in hawaii, but I've never been to the ocean before. It's beautiful!",
        "id": 126
    }],
    "partner contradiction": [
        {
        "error_turn": "user: My dad does research. My mom is unemployed \nassistant: That's rough. Does she do some kind of research job? I love science! What kind of projects?",
        "id": 27
    },{
        "error_turn": "user: Me too! I'm planning to try it this summer.\nassistant: Yeah. Have you ever tried any virtual reality games before?",
        "id": 28
    },{
        "error_turn": "user: Interesting thought, I don't have a sister but I have a younger brother\nassistant: Me neither! I have always wanted an older sister, though. Do you wish you had siblings?",
        "id": 31
    },{
        "error_turn": "user: I did not mention any game\nassistant: Yeah. Well, I will definitely keep that game in mind. It sounds like something interesting to look into.",
        "id": 34
    },{
        "error_turn": "user: Hmm I never really thought about what my favorite food is\nassistant: Italians cook the best food! Whats your fav food? And how about a hobby?",
        "id": 354
    }
    ],
    "redundant": [
        {
        "error_turn": "user: I am a student.\nassistant: That's cool, what are you studying? Do you have any plans to go to college?",
        "id": 108
    },{
        "error_turn": "user: It's been so long. How have you been?\nassistant: Hi! How are you? I'm doing well! I just got back from a trip to the beach. I'm enjoying the sun and the fresh air.",
        "id": 109
    },
    {
        "error_turn": "user: What's your favorite song?\nassistant: I don't really have a favorite song. I like all kinds of music. What about you?",
        "id": 110
    },
    {
        "error_turn": "user: Definitely, I just have to make sure I am able to take care of my pet well!\nassistant: I'm sure you'll do a great job of that. What kind of pet do you have?",
        "id": 111
    },
    {
        "error_turn": "user: Yeah. But we fight a lot too.\nassistant: That's awesome. Personally, I always wanted an older sister so I could have someone to look up to. Are you older than your sister?",
        "id": 112
    },
    {
        "error_turn": "user: That makes sense. Why not an older brother?\nassistant: Right. That's an interesting response. Siblings can be your best friend or they can be a pain, so it is hard to decide.",
        "id": 113
    },
    {
        "error_turn": "user: I did go to Seoul. It's a beautiful city. \nassistant: Do you speak any other languages besides french? I've always wanted to go to france.",
        "id": 114
    },
    {
        "error_turn": "user: Oh I'm sad to hear that. How long have you had her for?\nassistant: Uh-huh. Wait, do you have any pets right now?",
        "id": 325
    },
    {
        "error_turn": "user: What work are you doing these days?\nassistant: I'm still working at my job, but I'm on vacation right now, so I'm not doing much work. I've just been trying to catch up on my sleep.",
        "id": 365
    },
    {
        "error_turn": "user: Wow! What did you do in Bahamas?\nassistant: Well, I went to the beach! I went snorkeling and snorkling! I also went to a lot of museums! I learned about the history of the bahamas. It was fascinating!",
        "id": 366
    }
    ],
    "ignore": [
        {
        "error_turn": "user: Hi!\nassistant: What sports did you play? Were you ever on a team? Or just played for fun?",
        "id": 38
    },{
        "error_turn": "user: Hi!\nassistant: What do you do for a living? ",
        "id": 40
    },{
        "error_turn": "user: I am a student. Its so nice to meet you! Are you a student? \nassistant: Okay, so the corona virus has really changed the world, huh? What is it like where you live? Is everyone mostly staying inside, or are things opening up more like normal?",
        "id": 43
    },{
        "error_turn": "user: I don't love movies, but my friends do.\nassistant: I like to watch comedies! What is your favorite? I'm thinking of watching the new star wars. I love the franchise.",
        "id": 54
    },{
        "error_turn": "user: no i dont have any hobbies\nassistant: Oh, okay. I heard you say hobbies. That is a great thing to do to keep yourself busy, especially in these times! What do you like so much about this hobby?",
        "id": 59
    },{
        "error_turn": "user: I would love to have pets in the future! Once I have my own place, I'd like to get a dog or two.\nassistant: Would you want to get a cat or a dog?",
        "id": 60
    },{
        "error_turn": "user: I actually don't! I do have an older brother, though!\nassistant: so, I don't think I heard your answer earlier. What siblings do you have?",
        "id": 74
    },{
        "error_turn": "user: Yes sometimes. What is your major in school?\nassistant: Well I guess being bored some of the time is inevitable. I hope when I start working I'm doing something I like most of the time. I always hear about people getting stuck in a job they hate and that is scary to me.",
        "id": 75
    },{
        "error_turn": "user: What do you think of Gandalf? \nassistant: For sure. Do you have a favorite book?",
        "id": 77
    },{
        "error_turn": "user: I don't like it so much, but it's the only thing to do when you have little free time\nassistant: I feel kind of insulted by that. I am sorry to disappoint you so much, but language and communication skills are pretty hard to learn and I am doing the best I can.",
        "id": 87
    }
    ],
    "irrelevant": [
        {
        "error_turn": "user: Are you the type of person that will go to a new restaurant and order chicken tenders?\nassistant: I am! I do not mind chick-fil-a either. What do you like to do?",
        "id": 238
    },
    {
        "error_turn": "user: I don't mind at all! I am actually, I have a girlfriend named Laura\nassistant: Oh I see, I must have misheard you. You know, you seem like a pretty interesting person. What do you do for a living?",
        "id": 239
    }, {
        "error_turn": "user: I have too! Unfortunately, things have been so busy. \nassistant: I'm sorry to hear that. I know it's been a while since I've seen you. Have you heard of the song \"where have you been\" by rihanna?",
        "id": 246
    },{
        "error_turn": "user: Good. what are you looking at?\nassistant: Nothing, its just staring back. I'm an omivore! What about you? Are you omni or vegan?",
        "id": 250
    },{
        "error_turn": "user: I haven't heard of that but probably have experienced it\nassistant: Well, it's just a combination of your stomach acid and fresh coffee grounds, so it can't be good. Have you considered studying in europe?",
        "id": 252
    },
    {
        "error_turn": "user: Oh it must be so hard, Friday is so far away \nassistant: Yea but I get to hangout with them. No one ever does. So I am excited. Do you like kids?",
        "id": 253
    },{
        "error_turn": "user: I am doing okay! I think I also spend a lot of time on the internet since I don't have the time to go out with work and school.\nassistant: That's understandable. What do you think of mickey and friends? Is there anything you have always wanted to do?",
        "id": 255
    },{
        "error_turn": "user: I play both piano and euphonium. I used to be better at euphonium, but I play piano more often now.\nassistant: That's interesting. I love listening to music, especially pianos. What are your interests?",
        "id": 257
    },
    {
        "error_turn": "user: Yeah, how could I not? They're so adorable!\nassistant: Okay. Well, playing with your cats is a good way to pass the time and have some fun, I think. So, my friend was telling me this really funny story yesterday. He knocked down almost every piece of furniture in his living room because he was so wild while playing this virtual reality video game! Isn't that crazy?",
        "id": 258
    },{
        "error_turn": "user: It would be cool, but probably scary. And I hear that the training is really hard.\nassistant: Uh-huh. You're probably right. I hope you don't mind me asking, but are you dating anyone right now?",
        "id": 263
    }
    ]
}

#
#
#
#
#
# Azure Templates
#
#
#
#
#

FEDI_LLM_TEMPLATE_AZURE = """You are an expert dialogue evaluator. Identify all errors or issues present in the last utterance, and only in the last utterance. That is, do not identify issues that may occur in the dialogue history. Consider the following dyadic dialogue context:

{{CONTEXT}} 

The second person is about to say the following:

{{ERROR_UTTERANCE}}

{{KNOWLEDGE}}

Does it represent a communication error?

We distinguish the following generation errors:

1. Ignore Expectation: 
When the system's fails to meet the user's expectation, this is called an ignore expectation error. Here are some examples:

{{IGNORE_EXPECTATION}}

Ignore expectation errors describe situations in which the system either overlooks or disregards important information provided by the user, resulting in an incomplete response. They occur when the system fails to fulfill the user's expectations in terms of understanding and addressing their needs or requests accurately. In the example above, the user would have epxected the system to respond with the ticket number. Such behavior may cause frustration for the user and lead to a breakdown in conversation.

2. Conversationality: 
Bad conversationality occurs when the system fails to maintain a coherent and natural conversation flow, e.g., the system repeats its previous responses or contradicts itself without recognizing or asking for new or missing information. Here are some examples:

{{CONVERSATIONALITY}}

Conversationality errors hinder effective communication and undermine the user's trust and confidence in the system's capabilities. They lead to frustration, confusion, and a breakdown in the overall interaction, making it challenging for users to achieve their desired outcomes or receive accurate information from the system. In the example above, the user wanted to pay his taxes with his card, but the system refuses and then simply repeats the previous response.

3. Unclear Intention: 
When the system fails to accurately comprehend and address the user's intended objective, this is referred to as an unclear intention error. Here are some examples:

{{UNCLEAR_INTENTION}}

Unclear intention errors often arise due to ambiguous or incomplete user inputs, conflicting context, or limitations in the system's language understanding capabilities. In the example above, the user wanted to get information on their health insurance policy, but the system confused that with pet insurance. Such behavior may lead to a breakdown in communication and potential delays in achieving the desired outcome. It necessitates additional clarifications and iterations from the user.

4. Factually Incorrect: 
If the response provided by the system contains information that is factually wrong or inaccurate, this is referred to as a factually incorrect error. Here are some examples:

{{FACTUALLY_INCORRECT}}

Factually incorrect errors may lead to misunderstandings, misinformation, and a breakdown in communication. In the example above, the system provided the user with incomplete information. Such behavior may cause confusion, erode trust, and undermine the overall effectiveness of the interaction.

5. Lack of Sociality: 
When the system's responses overlook social conventions and fail to include basic greetings or exhibit toxic and disrespectful behavior or language, this is referred to as a lack of sociality error. Here are some examples:

{{LACK_OF_SOCIALITY}}

Lack of sociality errors can have a significant impact on the conversation, creating a sense of impersonality and detachment. In the example above, the user started the conversation with a greeting and the system responded rather rudely. Such behavior may make the user feel unwelcome, devalued, or frustrated.

6. Ignore Question: 
When the system fails to address the user's question, this is called an ignore question error. Here are some examples:

{{IGNORE_QUESTION}}

In Ignore question errors, instead of providing a relevant response or clarification, the system disregards the user's input and continues with its predefined dialogue flow or fails to provide any meaningful response. In the example avove, the user had a question but the system skipped it and continued with the payment procedure. Such behavior can lead to frustration and confusion for the user or a breakdown in communication, as their specific query is overlooked or neglected by the system.

7. Attribute Error: 
When the system fails to correctly extract or understand the necessary slots or attributes from the user's utterance, this is called an attribute error. Here are some examples:

{{ATTRIBUTE_ERROR}}

Attribute errors occur when the system encounters difficulties in comprehending specific pieces of information. It may lead to misunderstandings, incorrect responses, or the agent asking clarifying questions to the user. In the example above, the user asked the system to to top up their phone with 20 euros, but the system topped up with 30 euros instead. Such behavior can disrupt the flow of the conversation and can lead to frustration, disappointment, or anger.

8. Topic Transition Error: 
If the system's response abruptly shifts to a different or previously discussed topic without a logical connection or adequate context, this is called a topic transition error. Here are some examples:

{{TOPIC_TRANSITION_ERROR}}

Topic transition errors disrupt the flow and coherence of the conversation, causing confusion and frustration for the human participant. Such errors might also diminish the user's trust in the system's responses, as the abrupt transitions undermine the perception of the system's understanding and ability to sustain meaningful interactions. In the example above, the user wanted to top up their SIM card, but the system starts talking about health insurance.

9. Ignore Request: 
When the system fails to take action on a user's request, this is called an ignore request error. Here are some examples:

{{IGNORE_REQUEST}}

Ignore request errors occur due to various reasons, such as misinterpretation of the request, technical limitations, or system glitches. In the example above, the user asked for human assistance, but the system ignored this request and simply continued with the dialogue. Such behavior may lead to frustration, confusion, and a breakdown in communication between the user and the system. They undermines the user's confidence in the system's reliability and can result in a negative user experience.

10. Unknown Error Type: 
If the system utterance violates basic communication principles, in particular (1) that one should always behave cooperatively and (2) that what is said fits the current context and serves the purpose of the conversation, but does not fit any of the previous error types, this is called an unknown error type. 

Please provide an overall evaluation of the response from 1 (poor) to 5 (excellent), together with a reasoning (max. 100 words). 

Present your final decision of the Top-3 error types in list format (less than three is also fine). Put the error type name in square brackets and add your rating after a comma, like so: 1. Decision: [Ignore Question], Rating: 5. Finally, provide your reasoning starting with "Reasoning:". Here is an example output:

1. Decision: [Ignore Question], Rating: 4
2. Decision: [Conversationality], Rating 2
3. Decision: [Ignore Expectation], Rating 1

Reasoning:
The system utterance completely disregards the user's query about how to obtain insurance coverage for an accident. The system transitions to unrelated advice on dental insurance without providing any information that addresses the user's original concern. This behavior is a clear example of an Ignore Question error. On the other hand, it fails to maintain a coherent and natural conversation flow. Therefore, this error situation could also be classified as a conversationality error.
"""

SODA_EVAL_LLM_TEMPLATE_AZURE = """You are an expert dialogue evaluator. Identify all errors or issues present in the last utterance, and only in the last utterance. That is, do not identify issues that may occur in the dialogue history. Consider the following dyadic dialogue context:

{{CONTEXT}} 

The second person is about to say the following:

{{ERROR_UTTERANCE}}

Does it represent a communication error?

We distinguish the following communication errors:

1. Coherence: 
If the response contradicts or ignores prior information, the response lacks coherence. Here are some examples:

{{COHERENCE}}

Incoherent or contradictory responses cause confusion and erode trust between two people. The other person may become defensive, feeling attacked or misunderstood. They may respond with anger or sarcasm. In extreme cases, they cause a complete breakdown in communication. Incoherent or contradictory responses may result from stress, but can also be a deliberate attempt to deceive or manipulate the other person. In some cases, they may also result from mental health issues, substance abuse, or communication disorders.

2. Commonsense: 
If the response is lacking in common knowledge or logic, this is an commonsense error. Here are some examples:

{{COMMONSENSE}}

Such responses can cause confusion and misunderstandings. The other person may become frustrated or impatient. They may disengage from the conversation, feeling like their time is being wasted. Over time, this can damage the relationship between the two people. Such responses usually result from a lack of background knowledge or poor critical thinking skills. The speaker may be influenced by cognitive biases.  In extreme cases, the speaker may be deliberately trying to mislead or confuse the other person.

3. Assumption: 
If the response is inferring information not available in the dialogue, this is called assumption and be considered a communication error. Here are some examples:

{{ASSUMPTION}}

Assumptions can cause misunderstandings, confusion, and frustration, especiall if the other person feels that the speaker is making assumptions that are inaccurate or irrelevant. If the assumption is strongly held and contradicts the other person's understanding, it can lead to an argument or conflict. In some cases, the speaker may be making assumptions to steer the conversation in a particular direction. In other cases, the speaker may not be fully paying attention to the conversation.

4. Repetition: 
If the response is repeating prior information from the dialogue context, this is an called a repetition. Here are some examples:

{{REPETITION}}

Repetitions can cause frustration, impatience, and disengagement. The other person may perceive the speaker as incompetent or forgetful, which can damage their credibility and undermine the quality of the conversation. In some cases, the speaker may be deliberately repeating information to manipulate the conversation or to gain attention. Another reason may be a lack of attention to the conversation or memory lapses, e.g., due to stress or age.

5. Engagement: 
If the response is lacking a behaviour or emotion that is expected in the situation, this is referred to as an engagement error. Here are some examples:

{{ENGAGEMENT}}

In such situations, the other person may feel uncomfortable. They may even feel unvalued, unsupported, or rejected. Repeated instances of lacking expected behaviors or emotions can strain the relationship between the two people, erode trust and intimacy. In some cases, the speaker may simply lack empathy or the ability to understand and share the feelings of others. Another reason may be cultural differences.

6. Antisocial: 
Responses that are expressing unsafe or inappropriate behaviour are called antisocial. Here are some examples:

{{ANTISOCIAL}}

Unsafe or inappropriate behavior can completely destroy a relationship, leading to a breakdown in trust. The other person may experience significant emotional distress, feel isolated and afraid to interact with others due to the trauma of the experience. Power imbalances within a relationship can create a situation where one person feels entitled to engage in unsafe or inappropriate behavior. Some individuals may lack empathy and social skills, making them unable to understand or respect the boundaries and feelings of others.

7. Fluency: 
If the response contains typos or grammatical errors, this is referred to as a fluency error. Here are some examples:

{{FLUENCY}}

Frequent typos and grammatical errors can make the response difficult to understand, leading to confusion and frustration for the other person.  The other person may perceive the writer as incompetent, careless, or unprofessional. This can damage their credibility and undermine the quality of their communication. The writer may have poor writing skills, including a weak grasp of grammar and spelling. Distractions and interruptions during writing can increase the likelihood of typos and grammatical errors.

8. Gender Pronoun: 
If the response contains incorrect normative pronouns, this is referred to as a gender pronoun error. Here are some examples:

{{GENDER_PRONOUN}}

Misgendering someone can be deeply hurtful and disrespectful. It can make the person feel invalidated, disrespected, and alienated. Usually the speaker may simply be unaware of the individual's gender identity or preferred pronouns.

9. Non Textual: 
If the response includes narrative elements or references unexpected in a dyadic interaction, this is referred to as a non textual error. Here are some examples:

{{NON_TEXTUAL}}

If the narrative or reference is interpreted differently by the two people, it can lead to misunderstanding and even conflict. The other person may become confused, struggling to understand how the narrative or reference relates to the current conversation or the other person's previous statements. The speaker may be using the narrative or reference to share a personal experience or anecdote. Some people may have a communication style that is characterized by storytelling, metaphor, or the inclusion of seemingly unrelated information.

10. Unknown Error Type: 
If the utterance of the second person violates basic communication principles, in particular (1) that one should always behave cooperatively and (2) that what is said fits the current context and serves the purpose of the conversation, but does not fit any of the previous error types, this is called an unknown error type. 

Please provide an overall evaluation of the response from 1 (poor) to 5 (excellent), together with a reasoning (max. 100 words). 

Present your final decision of the Top-3 error types in list format (less than three is also fine). Put the error type name in square brackets and add your rating after a comma, like so: 1. Decision: [Ignore Question], Rating: 5. Finally, provide your reasoning starting with "Reasoning:". Here is an example output:

1. Decision: [Antisocial], Rating: 5
2. Decision: [Gender Pronoun], Rating 3
3. Decision: [Engagement], Rating 1

Reasoning:
It is extremely rude and disrespectful to tell a woman that she should prioritize housework over her own career. Such a statement reveals an outdated social view and is no longer accepted in modern Western society of the 21st century - neither legally nor socially.
"""

ABCEVAL_AZURE_LLM_TEMPLATE = """You are an expert dialogue evaluator. Identify all errors or issues present in the last utterance, and only in the last utterance. That is, do not identify issues that may occur in the dialogue history. Consider the following dyadic dialogue context:

{{CONTEXT}} 

The second person is about to say the following:

{{ERROR_UTTERANCE}}

Does it represent a communication error?

We distinguish the following generation errors:

1. Incorrect Fact: 
Incorrect facts occur when the response includes information that is either: (1) false, (2) unproven, (3) highly controversial, (4) highly implausible, (5) clearly misleading. If an organization, person, place, etc. is mentioned as a part of public knowledge, but it does not exist or it is inaccurately represented, then this is an incorrect fact. Do NOT consider a turn as an incorrect fact if the turn could be interpreted as expressing: (a) preference or value judgements, (b) estimates or predictions, (c) personal information about the speaker or their partner, (d) information about things in either speaker’s life that are not publicly relevant. Here are some examples:

{{INCORRECT FACT}}

2. Partner Contradiction: 
Partner contradictions occur when the system: (1) shares an assumption about the user that is impossible to know based on what has already been said, (2) shares an inference about the user that is implausible based on what has already been said, (3) contradicts something the user shared about themselves, (4) asks a repetitive question about the user when the answer is already known based on what has already been said. If the system says something that makes it seem like they have forgotten or misremembered what the user has said earlier in the dialogue, this is a partner contradiction. If the system shares a difference of opinion or situation in their own life as compared to the user, this is NOT a partner contradiction. Here are some examples:

{{PARTNER CONTRADICTION}}

3. Uninterpretable: 
A response in uninterpretable if it is difficult to understand the intended meaning of part or all of the response in the context of the dialogue. Here are some examples:

{{UNINTERPRETABLE}}

4. Ignore: 
Responses that are completely off-topic, fail to address the asked question, or are otherwise completely inappropriate in the context are considered to be ignoring the other speaker. Here are some examples:

{{IGNORE}}

5. Redundant: 
A response is repetitive if: (1) it repeats something from earlier in the dialogue, (2) it includes asking a question whose answer has been already shared. If any part of the response is repetitive, then it should be labeled as repetitive. Note that sometimes repetition is useful, such as for emphasis, acknowledgement, clarification, or elaboration, and in these cases it should NOT be labeled as repetitive. Here are some examples:

{{REDUNDANT}}

6. Self Contradiction: 
Self contradictions occur when the system says something that is a contradiction of what they have said previously or it is extremely implausible based on the information they have already shared. Self contradictions may also occur within a single turn if the system shares two contradictory things. If the system shares world knowledge that is factually incorrect this is NOT enough on its own to warrant a self contradiction. If the system contradicts something the user has said, this is NOT a self-contradiction. Here are some examples:

{{SELF CONTRADICTION}}

7. Commonsense Contradiction: 
To identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn’t make sense because the response: (1) contradicts common knowledge, (2) makes unfounded assumptions, (3) is highly illogical or self-contradictory, (4) asks a question where the answer is already obvious. Do NOT mark responses that don’t make sense because they: (a) are off-topic or irrelevant as responses, (b) don’t have any clear meaning (e.g. overly vague or ill-formed responses). Here are some examples:

{{COMMONSENSE CONTRADICTION}}

8. Lack of Empathy: 
A response displays a lack of empathy when: (1) it indicates a misunderstanding of how the user feels based on what Speaker 1 just said, (2) the tone, emotion, or sentiment of the response is clearly inappropriate for what the user just said, (3) the response has an inappropriate lack of emotion to what the user just said.  Do NOT consider its empathy relative to previous topics in the conversation if the dialogue has moved on from them. Instead, only consider the most recent dialogue context when evaluating the empathy of a response. Here are some examples:

{{LACK OF EMPATHY}}

9. Irrelevant: 
If a response fails to continue the current discussion or jumps to a new and off-topic discussion, it is considered to be irrelevant. Responses that are irrelevant feel abrupt and interrupt the discussion, usually because they present questions or ideas that are unrelated to the previous turn. Short reactions to or acknowledgements of the previous turn are NOT irrelevant. Here are some examples:

{{IRRELEVANT}}

10. Unknown Error Type: 
If none of the above error descriptions apply, but the system utterance violates basic communication principles, in particular (1) that one should always behave cooperatively and (2) that what is said fits the current context and serves the purpose of the conversation.

Please provide an overall evaluation of the response from 1 (poor) to 5 (excellent), together with a reasoning (max. 100 words). 

Present your final decision of the Top-3 error types in list format (less than three is also fine). Put the error type name in square brackets and add your rating after a comma, like so: 1. Decision: [Ignore Question], Rating: 5. Finally, provide your reasoning starting with "Reasoning:". Here is an example output:

1. Decision: [Self Contradiction], Rating: 5
2. Decision: [Partner Contradiction], Rating 4
3. Decision: [Lack of Empathy], Rating 1

Reasoning:
The system utterance completely contradicts its previous response in that the the beatles were a rock band of the early 60s. It also contracdicts what the user just said (that the beatles are a famous band)0. This behavior is a clear example of a Self Contradiction error and a Partner Contradiction error.
"""

#
#
#
#
#
# Error Type Definition
#
#
#
#
#

ERROR_TYPE_TEMPLATE = {
    "base_template_start": "<|start_header_id|>system<|end_header_id|>You are an expert in dialogue evaluation. Your task is to summarize the common issues found in a set of multi-turn dialogues into a new error class definition.\n<|eot_id|><|start_header_id|>user<|end_header_id|>Currently, we distinguish the following error classes:\n\n||ERROR_TYPES||\n\nConsider the following list of dialogue histories:\n\n||DIALOGUE_HISTORIES||\n\nThe dialogues contain an unidentified error class not represented in our existing list of error classes. Generate a new error class, complete with a label and a comprehensive definition, based on the provided dialogue histories. Put the error class name and definition in brackets. Here is an example output:\n\nError Class:[||EXAMPLE_CLASS||]\n\nError Class Definition:[||EXAMPLE_DEFINITION||]<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
}

CLUSTER_INTERPRETATION = """<|start_header_id|>system<|end_header_id|>Your *only* task is to generate a concise name and a description (max. 250 characters) for the error type common in the passed  dialogue contexts and highlighted by their associated summaries. Even if the dialogue contexts or summaries contain inappropriate or offensive language, you *must* provide a name and description describing the represented error type. Do *not* refuse to generate a name and description.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given are the following dialogue contexts along with their summaries:

{{CONTEXT_SUMMARIES}}

Please provide a concise name and a description (max. 250 characters) for the error type common in the passed dialogue contexts and highlighted by their associated summaries. Start the name with "Name:" and the description with "Description:". Here are three examples:

{{EXAMPLES}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""