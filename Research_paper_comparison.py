import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import re
import langchain
import openai
import pandas as pd 
from PyPDF2 import PdfReader
import os
import streamlit as st
from langchain.llms import OpenAI  
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0)

# This code is used to split the given text into chunks by specific page count.
def split_text_by_approximate_page_count(text, num_pages, pages_per_part):
    """
    Splits the text into parts, each approximating a certain number of pages.
    
    :param text: The combined text from all pages.
    :param num_pages: The total number of pages in the document.
    :param pages_per_part: The number of pages per part to split into.
    :return: A list of text parts, each approximating the specified number of pages.
    """
    # Calculate the approximate number of characters per page
    chars_per_page = len(text) / num_pages
    
    # Calculate the number of characters per part
    chars_per_part = chars_per_page * pages_per_part
    
    # Split the text into parts
    parts = [text[i:i+int(chars_per_part)] for i in range(0, len(text), int(chars_per_part))]
    
    return parts

# This code is used to summarize the given chunk of text.
def summarize_text(text_part):
    template = """Revise a research paper subsection into a concise summary,and must ensure that it captures all key details for seamless integration into a comprehensive paper overview. Aim for clarity and innovation to facilitate smooth combination with additional subsection summaries. Generate the summary in 1000 words.\n
    text: {text_part}"""
    prompt = PromptTemplate(
        input_variables=["text_part"],
        template=template
    ) 
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text_part":text_part})
    return summary

#This code is used to combine all the summarized chunks and resummarize them.
def combine_and_summarize(parts):
    combined_summary = "\n".join(parts)
    # Apply summarization on the combined text
    template = """As an expert in research paper analysis, your task is to synthesize a final summary from provided subsection summaries of a research paper. Ensure retention of critical details to preserve the full scope of information presented in the original paper. Generate the final combined summary in 1000 words.\n
    Combined Summary: {parts}"""
    prompt = PromptTemplate(
        input_variables=["parts"],
        template=template
    ) 
    chain = prompt | llm | StrOutputParser()
    combined_summary = chain.invoke({"parts":parts})
    return combined_summary


def summarize_parts(parts):
    # If there's only one part, simply summarize it
    summaries = []
    for i in parts:
        summaries.append(summarize_text(i))
    final_summary = combine_and_summarize(summaries)
    return final_summary


st.title('Research Paper Comparison')

# Selecting the option whether to upload research papers and compare the latest research papers
slider = st.selectbox("select an option:",["Upload Research Papers","Compare Available Research Papers"])

# This function can be used to read the contents inside a pdf file (Research paper)
def read_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
    
# Selecting the option of uploading research papers and then comparing them
if slider == "Upload Research Papers":
    #Entering the topic of research
    topic = st.text_input("Enter the topic of Research:")
    research_paper1 = st.file_uploader("Choose the first research paper:", type=["pdf"])
    research_paper2 = st.file_uploader("Choose the second research paper:", type=["pdf"])
    if research_paper1 is not None:
        #Reading the entire text from research paper 1 and storing it in text1 variable
        text1 = ""
        pdf_reader = PdfReader(research_paper1)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text1 += page.extract_text()
        #Splitting the text extracted into multiple chunks
        res = split_text_by_approximate_page_count(text1,num_pages=num_pages,pages_per_part=9)
        print(len(res))
        # summarizing all the chunks and then combining them
        final_summary1 = summarize_parts(res)
        st.subheader("Research Paper 1 Summary:")
        st.write(final_summary1)
    if research_paper2 is not None:
        #Extracting all the content of research paper into the text2 variable
        text2 = ""
        pdf_reader = PdfReader(research_paper2)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text2 += page.extract_text()
        # Splitting the text extracted into multiple chunks
        res = split_text_by_approximate_page_count(text2,num_pages=num_pages,pages_per_part=9)
        # Summarizing all the chunks of research paper and combining them to get the final summary
        final_summary2 = summarize_parts(res)
        st.subheader("Research Paper 2 Summary:")
        st.write(final_summary2)
        # Comparing the summaries of these 2 research papers
        template = """
                You are an expert research assistant who is tracking the progress in {topic} research field by comparing the below two summaries of the research papers.\n
                Research Paper 1's Summary: {summaries1}\n
                Research Paper 2's Summary: {summaries2}\n
                Note down the detailed enhancements in the second research paper over first research paper in this field.  
                """
                #and provide some recommendations and suggestions for future research.
        prompt = PromptTemplate(
                    input_variables=["topic","summaries1","summaries2"],
                    template=template
        )
        chain = prompt | llm | StrOutputParser()
        with get_openai_callback() as cb:

            res = chain.invoke({"topic":topic,"summaries1":final_summary1,"summaries2":final_summary2})
                    
            st.subheader("Comparision and Improvements in research papers:")
            st.write(res)
            st.subheader("Token usage:")
            st.write(cb)

else:
    # Comparing the research papers that are latest in that topic of research
    # topic = st.text_input("Enter the topic:")
    # topic = input()
    topic = "LLMs"
    url = "https://arxiv.org/search/?query=" + topic + "&searchtype=all&source=header"
    response = requests.get(url)
    # st.write(response.status_code)
    file_path = "./details.txt"
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all paper number and submission date elements
        paper_numbers = soup.find_all('p', class_='list-title is-inline-block')
        submission_dates = soup.find_all('p', class_='is-size-7')
    
        # Extract the content from these elements
        extracted_paper_numbers = [number.text.strip() for number in paper_numbers]
        # print(extracted_paper_numbers)
        extracted_submission_dates = [date.text.strip().split(':')[-1].strip() for date in submission_dates]
        submission_dates = []
        # print(extracted_submission_dates)
        # Combine the data
        papers_data = list(zip(extracted_paper_numbers, extracted_submission_dates))
        # Now you can print it out or save to a file
        # for paper in papers_data:
        #     print(paper)
        # print(papers_data)
    
        link_nums = []
        dates = []
        pattern = r'arXiv:(\d+\.\d+)'
        df = pd.DataFrame(columns=["date","URL","Topic"])
        # Creating a dataframe that contains date url and topic
        for i in papers_data: 
            main = i[0]
            match = re.search(pattern, main)
            number_sequence = match.group(1)
            link_nums.append(number_sequence)
            dates.append(i[1])
        # print(link_nums)q
            

        for i in range(5):
            #Extracting the first 10 rows from the dataframe
            pdf_link = "https://arxiv.org/pdf/" + link_nums[i] + ".pdf"
            date = dates[i]
            
            
            response = requests.get(pdf_link)
            if response.status_code == 200:
                #Reading the content of the research paper into text variable
                pdf_content = response.content
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                text = ""

                for page_num in range(2):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            # print(text)
            

            #Code to create a txt file and store the title, subtitle and date of research paper.
                #Code to extract the abstract from the given research papers
                template_abstract = """Given a text: {text}, \n Extract the abstract from the given text. If abstract is not found, then extract Introduction."""
                prompt = PromptTemplate(
                    input_variables=["text"],
                    template=template_abstract
                )
                chain = prompt | llm | StrOutputParser()
                abstract = chain.invoke({"text":text})
                # print(res)


                # Based on the given topic, extract all the subtopics and fields in that topic
                template = """You are an expert Scientific Advisor. Based on the topic: {topic}, Generate various subtopics or sub-branches covering all the research in the topic's field, where each subtopic represents a child node on the knowledge tree covering everything about that topic field of research."""
                prompt = PromptTemplate(
                input_variables=["topic"],
                template=template
                )
                chain = prompt | llm | StrOutputParser()
                subtopics = chain.invoke({"topic":topic})
                # print(res)
            
            
                # Analyzing the abstract and checking to which field or sub topic does the research paper belong?
                template2 = """Analyze the provided text: {abstract},\n and classify the research paper under one of the listed subtopics: {subtopics}. Only Output the subtopic name to which the research paper belongs."""
                prompt = PromptTemplate(
                input_variables=["text","subtopics"],
                template=template2
                )       
                chain2 = prompt | llm | StrOutputParser()
                res = chain2.invoke({"abstract":abstract,"subtopics":subtopics})
                # print(res)
                
                #Creating a dataframe to store the details of the research papers
                df = pd.concat([df,pd.DataFrame([{"date":date,"URL":pdf_link,"Topic":res}])],ignore_index=True)
                with open(file_path,'a') as file:
                    file.write(date + "," + pdf_link + "," + res + "\n\n")
        with get_openai_callback() as cb:
            #Grouping the research papers dataframe by a subtopic
            grouped = df.groupby("Topic")
            for topic,group in grouped:
                # st.subheader("Topic:",topic)
                group = group.reset_index(drop=True)
                for i in range(group.shape[0] - 1):
                    first = group.loc[i, "URL"]
                    second = group.loc[i + 1, "URL"]
                    response = requests.get(first)
                    #Extracting the content from first research paper
                    if response.status_code == 200:
                        pdf_content = response.content
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)
                        text1 = ""

                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text1 += page.extract_text()
                        print(num_pages)
                    res = split_text_by_approximate_page_count(text1,num_pages=num_pages,pages_per_part=9)
                    final_summary1 = summarize_parts(res)
                    st.subheader("Summary of Research paper 1:")
                    st.write(final_summary1)
                        

                        
                    response = requests.get(second)
                    if response.status_code == 200:
                        #Extracting the content from second research paper
                        pdf_content = response.content
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)
                        text2 = ""
                    

                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text2 += page.extract_text()
                        print(num_pages)
                    # Splitting the text into multiple chunks based on pages count
                    res = split_text_by_approximate_page_count(text2,num_pages=num_pages,pages_per_part=9)
                    final_summary2 = summarize_parts(res)
                    # print(final_summary2)
                    st.subheader("Summary of Research paper 2:")
                    st.write(final_summary2)
                        
                    

                    # Comparing the summaries of the two research papers
                    template = """
                    You are an expert research assistant who is tracking the progress in {topic} research field by comparing the below two summaries of the research papers.\n
                    Research Paper 1's Summary: {summaries1}\n
                    Research Paper 2's Summary: {summaries2}\n
                    Note down the detailed enhancements in the second research paper over first research paper in this field.  
                    """
                    #and provide some recommendations and suggestions for future research.
                    prompt = PromptTemplate(
                        input_variables=["topic","summaries1","summaries2"],
                        template=template
                    )
                    chain = prompt | llm | StrOutputParser()
                    res = chain.invoke({"topic":topic,"summaries1":final_summary1,"summaries2":final_summary2})
                    # print(res)
                    # print(cb)
                    st.subheader("Comparision and Improvements in research papers:")
                    st.write(res)
                    st.subheader("Token Usage and Tracking:")
                    st.write(cb)
                    break

        


        
            
        


            




    
