from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

d1='''
Sure, here is a prompt for cold emailing for jobs:

**Subject:** [Your Name] - Excited about [Position] at [Company Name]

**Body:**

Dear [Hiring Manager Name],

I hope this email finds you well.

I am writing to express my keen interest in the [Position] opening at [Company Name]. I have been following [Company Name]'s work for some time now and am consistently impressed by your innovative approach to [Industry] and your commitment to [Company Values].

I am particularly drawn to this opportunity because of my experience in [Relevant Experience]. In my previous role at [Previous Company], I was responsible for [List of Responsibilities]. I have a proven track record of [List of Achievements].

I am confident that my skills and experience would be a valuable asset to [Company Name]. I am a highly motivated and results-oriented individual with a passion for [Industry]. I am also a quick learner and have a proven ability to adapt to new challenges.

I am eager to learn more about the [Position] and how I can contribute to [Company Name]'s continued success. I am available for an interview at your earliest convenience.

Thank you for your time and consideration.

Sincerely,
[Your Name]'''
d2='''
Sure, here is a prompt for cold emailing for jobs:

**Subject: Passionate [Your Job Title] Eager to Contribute to [Company Name]'s Success**

**Dear [Hiring Manager Name],**

I hope this email finds you well.

My name is [Your Name], and I am writing to express my keen interest in the [Job Title] position currently advertised at [Company Name]. With my [Number] years of experience in the [Industry] industry and a proven track record of success in [Specific Achievements], I am confident that I possess the skills and expertise to make significant contributions to your team.

Throughout my career, I have demonstrated a strong ability to [List Key Skills and Accomplishments]. I am also a highly motivated and results-oriented individual with a passion for [Industry] and a commitment to excellence.

I am particularly interested in working at [Company Name] because of your company's reputation for innovation and commitment to [Company Values]. I am also impressed by your company's recent [Company Achievements], and I am eager to be a part of a team that is making a real difference in the [Industry] industry.

I have attached my resume for your review, which provides further details of my qualifications and experience. I am also available for an interview at your earliest convenience.

Thank you for considering my application. I look forward to hearing from you soon.

Sincerely,
[Your Name]'''
def compare (a,b):
  return cosine_similarity(model.encode([a]),model.encode([b]))

