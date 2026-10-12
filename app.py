"""
TextAnatomy - an NLP Pipeline Visualizer
Paste any text and watch it move through every stage of classical
and modern NLP: morphology, syntax, semantics, pragmatics, and
applications. Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from modules import applications, morphology, pragmatics, semantics, syntax_analysis

st.set_page_config(page_title="TextAnatomy - NLP Pipeline Visualizer", layout="wide")

SAMPLE_TEXT = (
    "Ravi went to the market to buy vegetables. He met his friend Suresh "
    "there, and they decided to have lunch together. Suresh said he was "
    "very hungry because he had skipped breakfast. After lunch, Ravi "
    "returned home and cooked dinner for his family. He was happy that "
    "the meal turned out well."
)

st.title("TextAnatomy")
st.caption(
    "Paste any text and watch it move through every stage of NLP: "
    "morphology, syntax, semantics, pragmatics, and applications."
)

text_input = st.text_area("Enter text to analyze", value=SAMPLE_TEXT, height=180)

if st.button("Analyze", type="primary"):
    st.session_state["analyzed"] = True
    st.session_state["text"] = text_input

if st.session_state.get("analyzed"):
    text = st.session_state["text"]

    if len(text.split()) < 5:
        st.warning("Please enter a longer piece of text (at least a few sentences).")
        st.stop()

    tabs = st.tabs(
        [
            "Overview",
            "Morphology",
            "Syntax",
            "Semantics",
            "Pragmatics",
            "Applications",
        ]
    )

    # ---- Overview ---------------------------------------------------
    with tabs[0]:
        st.subheader("Overview")
        st.write(
            "This dashboard mirrors the classic NLP pipeline: text is "
            "broken into words (morphology), structured into grammar "
            "(syntax), interpreted for meaning (semantics), linked "
            "across sentences (pragmatics/discourse), and finally used "
            "for real applications like summarization and Q&A."
        )
        col1, col2 = st.columns(2)
        col1.metric("Word count", len(text.split()))
        col2.metric("Character count", len(text))

    # ---- Module 2: Morphology ----------------------------------------
    with tabs[1]:
        st.subheader("Module 2: Morphological Analysis")
        try:
            result = morphology.analyze_morphology(text)
            st.write(f"Total words: {result['word_count']}  |  Unique words: {result['unique_word_count']}")
            st.markdown("**Stemming (Porter) vs Lemmatization (WordNet)**")
            st.dataframe(result["comparison_table"], use_container_width=True)
            st.markdown("**Most frequent bigrams**")
            st.dataframe(result["ngram_table"], use_container_width=True)
        except Exception as exc:
            st.error(f"Morphology analysis failed: {exc}")

    # ---- Module 3: Syntax ---------------------------------------------
    with tabs[2]:
        st.subheader("Module 3: Syntactic Analysis")
        try:
            st.markdown("**POS tagging (spaCy)**")
            pos_tags = syntax_analysis.spacy_pos_tagging(text)
            df_pos = pd.DataFrame(pos_tags, columns=["Token", "POS (universal)", "Tag (detailed)"])
            st.dataframe(df_pos, use_container_width=True)
        except Exception as exc:
            st.error(f"spaCy POS tagging failed: {exc}")

        try:
            st.markdown("**Toy HMM POS tagger (trained on Brown corpus)**")
            with st.spinner("Training HMM tagger on a Brown corpus sample (first run only)..."):
                tagger = syntax_analysis.get_hmm_tagger()
            first_sentence = text.split(".")[0].split()
            if first_sentence:
                hmm_tags = tagger.tag(first_sentence)
                st.dataframe(pd.DataFrame(hmm_tags, columns=["Word", "HMM Tag"]), use_container_width=True)
        except Exception as exc:
            st.error(f"HMM tagger failed: {exc}")

        try:
            st.markdown("**CFG parse tree (fixed demo sentence, shows structural ambiguity)**")
            sentence, trees = syntax_analysis.get_cfg_parse()
            st.code(" ".join(sentence))
            if not trees:
                st.info("No parse found for the demo sentence with this grammar.")
            for i, tree in enumerate(trees, 1):
                st.text(f"Parse {i}:\n{tree.pformat()}")
        except Exception as exc:
            st.error(f"CFG parsing failed: {exc}")

    # ---- Module 4: Semantics -------------------------------------------
    with tabs[3]:
        st.subheader("Module 4: Semantic Analysis")
        try:
            ambiguous_words = semantics.find_ambiguous_words(text)
            st.write("Automatically detected ambiguous words (multiple WordNet senses):")
            st.write(", ".join(ambiguous_words) if ambiguous_words else "None found.")
            if ambiguous_words:
                st.markdown("**Lesk algorithm - word sense disambiguation in context**")
                wsd_results = semantics.wsd_analysis(text, ambiguous_words)
                st.dataframe(pd.DataFrame(wsd_results), use_container_width=True)
        except Exception as exc:
            st.error(f"Semantic analysis failed: {exc}")

    # ---- Module 5: Pragmatics -------------------------------------------
    with tabs[4]:
        st.subheader("Module 5: Pragmatics and Discourse")
        try:
            st.markdown("**Coreference resolution (coreferee)**")
            chains = pragmatics.get_coreference_chains(text)
            if chains:
                for i, chain in enumerate(chains, 1):
                    st.write(f"Chain {i}: " + " -> ".join(chain))
            else:
                st.info(
                    "coreferee found no chains (or is not installed). "
                    "Showing a simple recency-based heuristic instead."
                )
                heuristic = pragmatics.simple_pronoun_resolution(text)
                if heuristic:
                    st.dataframe(pd.DataFrame(heuristic), use_container_width=True)
                else:
                    st.write("No pronouns with a preceding noun were found.")
        except Exception as exc:
            st.error(f"Pragmatics analysis failed: {exc}")

    # ---- Module 6: Applications -----------------------------------------
    with tabs[5]:
        st.subheader("Module 6: Applications")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Summarization**")
            try:
                with st.spinner("Summarizing..."):
                    summary = applications.summarize_text(text)
                st.success(summary)
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")

            st.markdown("**Sentiment**")
            try:
                sentiment = applications.analyze_sentiment(text)
                st.write(f"{sentiment['label']}  (confidence {sentiment['score']:.2f})")
            except Exception as exc:
                st.error(f"Sentiment analysis failed: {exc}")

        with col2:
            st.markdown("**Named entities**")
            try:
                entities = applications.extract_entities(text)
                if entities:
                    st.dataframe(pd.DataFrame(entities), use_container_width=True)
                else:
                    st.write("No named entities found.")
            except Exception as exc:
                st.error(f"NER failed: {exc}")

            st.markdown("**Ask a question about the text**")
            question = st.text_input("Your question", key="qa_question")
            if question:
                try:
                    answer = applications.answer_question(text, question)
                    st.write(f"Answer: **{answer['answer']}**  (confidence {answer['score']:.2f})")
                except Exception as exc:
                    st.error(f"Question answering failed: {exc}")

        st.markdown("**Search within the text (TF-IDF information retrieval)**")
        query = st.text_input("Search query", key="ir_query")
        if query:
            try:
                hits = applications.find_relevant_sentences(text, query)
                for sentence, score in hits:
                    st.write(f"({score:.2f}) {sentence}")
            except Exception as exc:
                st.error(f"Search failed: {exc}")
