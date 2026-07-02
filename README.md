# 1001 Occupations - Singapore's SSOC 2024 Based Career Matcher

## Overview

The 1001 Occupations app is an interactive career exploration and job matching platform built on Singapore's Standard Occupational Classification (SSOC) 2024 framework. It enables users to discover new career paths, evaluate job opportunities, and find matches based on their professional profile and resume.

## What It Does

- **Occupation Discovery**: Browse through all 1001 Singapore occupations and explore their detailed responsibilities, required skills, salary ranges, and career contexts
- **Semantic Search**: Find occupations and jobs based on natural language queries, matching on job responsibilities and career requirements
- **Personalized Job Matching**: Compare job postings against your profile (resume, target occupation, career direction) to find the best fit
- **Career Path Analysis**: Visualize semantic relationships between occupations and discover similar career options

## Key Features

1. **Comprehensive Occupational Data**: Access detailed Occupation information from the Singapore SSOC 2024 framework
2. **Intelligent Matching**: AI-powered semantic search and profile matching against real job postings
3. **Visual Career Exploration**: Interactive occupational maps and skill relationship visualizations
4. **Personal Career Planning**: Set target occupations, career direction, and analyze resume alignment

## Data Sources

- **Main Data**: Singapore Standard Occupational Classification 2024 (SSOC 2024) - occupational definitions, tasks, skills, and career hierarchies
- **Job Market**: Real job posting data from MyCareersFuture.sg and other Singapore job boards
- **Semantic Analysis**: Pre-computed vector embeddings enabling fast similarity search

## How to Use

1. **Get Started**: Upload or paste your resume in the "My Profile" section
2. **Find Your Fit**: Explore occupation matches or search by job title
3. **Set Goals**: Choose a target occupation and career direction (Exact Match, Career Fit, Balanced, Career Pivot, or Career Transition)
4. **Discover Jobs**: Find matching job opportunities based on your profile
5. **Bring Your Own Job**: Evaluate any job posting against your profile

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: Transformers, Sentence transformers, Gradio 
- **Visualization**: Plotly, custom interactive charts
- **Data Processing**: Pandas, NumPy

## Notes

- This app was built using opencode
- Job data is not updated currently