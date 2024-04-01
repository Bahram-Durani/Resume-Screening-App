import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb')) 

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.title("Welcome To The Resume Screening App")
    uploaded_file = st.file_uploader('Please Upload Your Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)
        
        
        # GitHub link with logo and name at the bottom left side
        st.markdown(
                """
                <style>
                .github-container {
                    position: fixed;
                    bottom: 10px;
                    left: 10px;
                    display: flex;
                    align-items: center;
                }
                .github-logo {
                    width: 50px;
                    margin-right: 10px;
                }
                .github-name {
                    font-size: 16px;
                }
                </style>
                """
                , unsafe_allow_html=True
            )

        st.markdown(
                '<div class="github-container">'
                '<span class="github-name">Get in touch with Bahram Durani: </span>'
                '<a href="https://www.linkedin.com/in/bahram-durani-b46876237/" target="_blank"><img src=" https://th.bing.com/th/id/OIP.QkU0Vf5aO5Gv8Yf4rXc4qwHaHa?w=191&h=191&c=7&r=0&o=5&dpr=1.3&pid=1.7w" class="github-logo"></a>'

                '<a href="https://github.com/Bahram-Durani" target="_blank"><img src="https://th.bing.com/th/id/OIP.8SVgggxQcO5L6Dw_61ac4QHaEK?w=260&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7" class="github-logo"></a>'

                '</div>'
                , unsafe_allow_html=True
            )




# python main
if __name__ == "__main__":
    main()
