import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# LOAD ENVIRONMENT
# =====================================================

load_dotenv()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


if not GROQ_API_KEY:
    st.error(
        "Groq API Key not found! Please add it to .env file."
    )
    st.stop()



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="AI Loan Advisory Assistant",

    layout="wide"

)



# =====================================================
# EMI CALCULATOR
# =====================================================

def calculate_emi(
        principal,
        annual_rate,
        years
):

    monthly_rate = annual_rate / (12 * 100)

    months = years * 12


    if monthly_rate == 0:

        emi = principal / months


    else:

        emi = (

            principal
            *
            monthly_rate
            *
            (1 + monthly_rate) ** months

        ) / (

            ((1 + monthly_rate) ** months) - 1

        )


    total_payment = emi * months

    total_interest = total_payment - principal


    return emi, total_interest, total_payment




# =====================================================
# LOAN ELIGIBILITY CHECKER
# =====================================================

def check_loan_eligibility(

        income,

        existing_emi,

        age,

        loan_amount,

        rate,

        tenure

):


    if age < 21:

        return {

            "eligible": False,

            "message":
            "Applicant age should be minimum 21 years."

        }




    if age + tenure > 70:

        return {

            "eligible": False,

            "message":
            "Loan tenure exceeds maximum age limit."

        }




    emi, _, _ = calculate_emi(

        loan_amount,

        rate,

        tenure

    )



    total_obligation = existing_emi + emi



    foir = (

        total_obligation / income

    ) * 100




    if foir <= 50:


        return {

            "eligible": True,

            "message":

            f"Loan appears eligible. "
            f"FOIR is {foir:.2f}% "
            "which is within acceptable limits.",


            "emi": emi

        }



    else:


        return {

            "eligible": False,

            "message":

            f"Loan may not be eligible. "
            f"FOIR is {foir:.2f}% "
            "which is above acceptable limits.",


            "emi": emi

        }





# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:


    st.title("AI Loan Assistant")


    st.markdown("""

This AI assistant answers loan queries using uploaded documents.

### Tech Stack

- LangChain
- FAISS
- HuggingFace Embeddings
- Groq Llama 3.3
- Streamlit

""")


    st.divider()



    # ==========================
    # EMI CALCULATOR
    # ==========================

    st.subheader("💰 EMI Calculator")


    loan_amount = st.number_input(

        "Loan Amount (₹)",

        min_value=10000,

        value=5000000,

        step=10000

    )


    interest_rate = st.number_input(

        "Interest Rate (%)",

        min_value=0.1,

        value=8.5,

        step=0.1

    )


    tenure_years = st.number_input(

        "Loan Tenure (Years)",

        min_value=1,

        value=20

    )



    if st.button("Calculate EMI"):


        emi, interest, total = calculate_emi(

            loan_amount,

            interest_rate,

            tenure_years

        )


        st.success("EMI Details")


        st.write(
            f"Monthly EMI: ₹{emi:,.2f}"
        )


        st.write(
            f"Total Interest: ₹{interest:,.2f}"
        )


        st.write(
            f"Total Payment: ₹{total:,.2f}"
        )



    st.divider()



    # ==========================
    # ELIGIBILITY CHECKER
    # ==========================

    st.subheader("Loan Eligibility Checker")


    income = st.number_input(

        "Monthly Income (₹)",

        min_value=10000,

        value=60000,

        step=5000

    )


    existing_emi = st.number_input(

        "Existing EMI (₹)",

        min_value=0,

        value=0,

        step=1000

    )


    age = st.number_input(

        "Applicant Age",

        min_value=18,

        value=30

    )


    required_amount = st.number_input(

        "Required Loan Amount (₹)",

        min_value=100000,

        value=4000000,

        step=100000

    )


    eligibility_rate = st.number_input(

        "Interest Rate (%)",

        value=8.5

    )


    eligibility_tenure = st.number_input(

        "Tenure (Years)",

        min_value=1,

        value=20

    )



    if st.button("Check Eligibility"):


        result = check_loan_eligibility(

            income,

            existing_emi,

            age,

            required_amount,

            eligibility_rate,

            eligibility_tenure

        )



        if result["eligible"]:

            st.success(
                "✅ Eligible"
            )

        else:

            st.error(
                "❌ Not Eligible"
            )



        st.write(
            result["message"]
        )


        if "emi" in result:

            st.write(

                f"Estimated EMI: ₹{result['emi']:,.2f}"

            )



# =====================================================
# MAIN APPLICATION
# =====================================================


st.title(
    "AI Loan Advisory Assistant"
)


st.caption(
    "Ask questions related to home loans using uploaded loan documents."
)



# =====================================================
# LOAD EMBEDDINGS
# =====================================================


embeddings = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"

)



if not os.path.exists("vector_db"):

    st.error(
        "Vector database not found. Run create_vector_db.py first."
    )

    st.stop()



db = FAISS.load_local(

    "vector_db",

    embeddings,

    allow_dangerous_deserialization=True

)



retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
    "k":6,
    "fetch_k":20
    }
)



# =====================================================
# LLM
# =====================================================


llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="llama-3.3-70b-versatile",

    temperature=0

)
# =====================================================
# ANSWER GENERATION PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template("""

You are an AI Loan Advisory Assistant.

Answer questions ONLY from the provided loan document context.

Rules:

- Do not use outside knowledge.
- Do not guess or assume.
- Do not add information that is not present in the context.
- If the answer is not available, reply exactly:

"I couldn't find this information in the provided loan documents."

- Combine information from multiple chunks if required.
- Mention exact values from documents.
- Keep answers clear and structured.
- Use bullet points for fees, charges, eligibility, and rules.


Context:

{context}


Question:

{question}


Answer:

""")



# =====================================================
# RESPONSE VALIDATION PROMPT
# =====================================================

validation_prompt = ChatPromptTemplate.from_template("""

You are a response validator for an AI Loan Advisory Assistant.

Your task is to check whether the generated answer is supported by the retrieved loan document context.

Rules:

- If the answer is completely supported by the provided context, reply only:

VALID


- Reply only:

INVALID

if the answer:
  - contains information not present in the context
  - gives incorrect values
  - makes assumptions or guesses
  - uses outside knowledge
  - contradicts the provided context


Do not mark an answer INVALID just because the wording is different from the context.
The answer can be considered VALID if it correctly summarizes or explains information from the context.


Context:

{context}


Generated Answer:

{answer}


Validation Result:

""")



# =====================================================
# CREATE CHAINS
# =====================================================

answer_chain = prompt | llm

validation_chain = validation_prompt | llm




# =====================================================
# USER QUESTION
# =====================================================

question = st.text_input(

    "Ask a question",

    placeholder=
    "Example: What is the processing fee?"

)



if question:


    with st.spinner(
        "Searching loan documents..."
    ):


        # Retrieve documents

        search_question = question
        if "processing fee" in question.lower():
            search_question += " salaried customer 0.35% GST maximum minimum charges"
        docs = retriever.invoke(search_question)
        
        docs = docs
      

        if not docs:

            st.error(
                "No relevant information found."
            )

            st.stop()



        # Combine retrieved chunks

        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )



        # -----------------------------
        # Generate Answer
        # -----------------------------

        response = answer_chain.invoke({

            "context": context,

            "question": question

        })



        # -----------------------------
        # Validate Answer
        # -----------------------------

        validation = validation_chain.invoke({

            "context": context,

            "answer": response.content

        })




        if "INVALID" in validation.content.upper():

            final_answer = (

                "I couldn't verify this information "

                "from the provided loan documents."

            )


            validation_status = False



        else:


            final_answer = response.content

            validation_status = True





    # =====================================================
    # DISPLAY ANSWER
    # =====================================================


    st.subheader(
        "Answer"
    )


    st.info(
        final_answer
    )



    # =====================================================
    # VALIDATION RESULT
    # =====================================================


    with st.expander(
        "Response Validation"
    ):


        if validation_status:


            st.success(
                "✅ Response verified against retrieved documents."
            )


        else:


            st.warning(
                "⚠️ Response failed verification."
            )




    # =====================================================
    # SOURCE DOCUMENTS
    # =====================================================


    st.subheader(
        "Source Documents"
    )


    shown = set()



    for doc in docs:


        source = os.path.basename(

            doc.metadata.get(

                "source",

                "Unknown"

            )

        )


        page = (

            doc.metadata.get(

                "page",

                0

            )

            + 1

        )



        if source not in shown:


            shown.add(source)


            st.write(

                f"📄 {source} (Page {page})"

            )




    # =====================================================
    # DEVELOPER CONTEXT
    # =====================================================


    with st.expander(

        "Retrieved Context (Developer Only)"

    ):


        for i, doc in enumerate(

            docs,

            start=1

        ):


            st.markdown(

                f"### Chunk {i}"

            )


            st.write(

                f"**Source:** "
                f"{os.path.basename(doc.metadata.get('source','Unknown'))}"

            )


            st.write(

                f"**Page:** "
                f"{doc.metadata.get('page',0)+1}"

            )


            st.text(

                doc.page_content

            )


            st.divider()





# =====================================================
# FOOTER
# =====================================================


st.markdown("---")


st.caption("""

⚠️ Responses are generated only from uploaded loan documents
and should not be considered financial or legal advice.

""")