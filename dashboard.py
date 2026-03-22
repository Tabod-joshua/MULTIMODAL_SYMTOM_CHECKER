import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px

st.set_page_config(page_title="DermLIP Evaluation Dashboard", layout="wide", page_icon="🏥")

st.title("🏥 DermLIP Evaluation Dashboard")
st.markdown("Visual analysis of the latest evaluation reports.")

REPORT_DIR = "eval_reports"

def load_latest_file(pattern):
    if not os.path.exists(REPORT_DIR):
        return None
    files = glob.glob(os.path.join(REPORT_DIR, pattern))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_json(filepath):
    if not filepath:
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load data
class_file = load_latest_file("classify_*.json")
class_csv_file = load_latest_file("classify_*.csv")
kb_file = load_latest_file("kb_coverage_*.json")
latency_file = load_latest_file("full_eval_*.json") 
chat_file = load_latest_file("chat_quality_*.json")
llm_file = load_latest_file("llm_polish_*.json")

class_data = load_json(class_file)
kb_data = load_json(kb_file)
full_eval_data = load_json(latency_file)
chat_data = load_json(chat_file)
llm_data = load_json(llm_file)

if not os.path.exists(REPORT_DIR):
    st.error(f"The directory '{REPORT_DIR}' does not exist. Please run evaluate.py first.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Classification Accuracy", 
    "📚 KB Coverage", 
    "⚡ Latency", 
    "💬 Chat Quality",
    "🤖 LLM Polish"
])

# ==========================================
# TAB 1: CLASSIFICATION
# ==========================================
with tab1:
    st.header("Classification Accuracy (Module 1)")
    if class_data:
        st.info(f"Loaded: `{os.path.basename(class_file)}`")
        
        top1_acc = 0
        topk_acc = 0
        kb_hit = 0
        total = 0
        
        if full_eval_data and 'classify' in full_eval_data:
            c = full_eval_data['classify']
            top1_acc = c.get('top1_accuracy', 0)
            topk_keys = [k for k in c.keys() if k.startswith('top') and k.endswith('_accuracy') and k != 'top1_accuracy']
            topk_key = topk_keys[0] if topk_keys else 'top3_accuracy'
            topk_acc = c.get(topk_key, 0)
            kb_hit = c.get('kb_hit_rate_pct', 0)
            total = c.get('total_images', 0)
        else:
            total = len(class_data)
            if total > 0:
                t1 = sum(1 for r in class_data if r.get('top1_correct'))
                tk = sum(1 for r in class_data if r.get('topk_correct'))
                kbh = sum(1 for r in class_data if r.get('kb_hit'))
                top1_acc = (t1 / total) * 100
                topk_acc = (tk / total) * 100
                kb_hit = (kbh / total) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Overall Top-1 Accuracy", 
            f"{top1_acc:.1f}%",
            help="Percentage of images where the model's highest-confidence prediction exactly matched the true condition."
        )
        col2.metric(
            "Overall Top-K Accuracy", 
            f"{topk_acc:.1f}%",
            help="Percentage of images where the true condition was in the model's top 3 predictions."
        )
        col3.metric(
            "KB Hit Rate", 
            f"{kb_hit:.1f}%",
            help="Percentage of images where the model's top prediction exists in the knowledge base, ensuring the LLM can generate a safe, informed response."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        col4.metric(
            "Total Images Evaluated", 
            total,
            help="Total number of images run through the classification model."
        )
        
        total_correct = int(round((top1_acc / 100) * total))
        total_incorrect = total - total_correct
        
        col5.metric(
            "Total Correct Predictions", 
            total_correct,
            help="Absolute number of correctly diagnosed images (Top-1)."
        )
        col6.metric(
            "Total Incorrect Predictions", 
            total_incorrect,
            help="Absolute number of incorrectly diagnosed images (Top-1)."
        )
        
        st.markdown("---")
        st.subheader("Performance by Condition")
        
        csv_path = class_file.replace('.json', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Group by true condition label and calculate accuracies
            df['top1_correct'] = df['top1_correct'].astype(bool)
            df['topk_correct'] = df['topk_correct'].astype(bool)
            
            # Group by true condition label and calculate accuracies (Total Images sent in and Hits)
            perf_by_cond = df.groupby('label').agg(
                Total_Images=('image', 'count'),
                Top1_Hits=('top1_correct', 'sum'),
                TopK_Hits=('topk_correct', 'sum')
            ).reset_index()
            
            # Calculate Derived Metrics
            perf_by_cond['Top-1 Accuracy'] = perf_by_cond['Top1_Hits'] / perf_by_cond['Total_Images']
            perf_by_cond['Top-K Accuracy'] = perf_by_cond['TopK_Hits'] / perf_by_cond['Total_Images']
            perf_by_cond['Total_Incorrect'] = perf_by_cond['Total_Images'] - perf_by_cond['Top1_Hits']
            
            # Sort by count so biggest classes are first
            perf_by_cond = perf_by_cond.sort_values('Total_Images', ascending=False)
            
            # Bar Chart (All Evaluated Conditions)
            fig = px.bar(
                perf_by_cond, 
                x='label', 
                y=['Top-1 Accuracy', 'Top-K Accuracy'], 
                title=f"Accuracy for All Evaluated Conditions ({len(perf_by_cond)} total)", 
                barmode='group',
                labels={'value': 'Accuracy Rate', 'label': 'Condition', 'variable': 'Metric'}
            )
            fig.update_layout(
                yaxis_tickformat='.0%',
                xaxis_title="Condition",
                yaxis_title="Accuracy",
                width=max(800, len(perf_by_cond) * 35),
                height=500
            )
            # Do not use container width so it enforces the wide figure and horizontal scroll
            st.plotly_chart(fig)
            
            # Data Table (All)
            st.markdown("### Detailed Accuracy (All Conditions)")
            
            display_perf = perf_by_cond[['label', 'Total_Images', 'Top1_Hits', 'Total_Incorrect', 'Top-1 Accuracy', 'Top-K Accuracy']].copy()
            display_perf.columns = ['Condition', 'Total Input Images', 'Correct Predictions', 'Incorrect Predictions', 'Top-1 Accuracy', 'Top-K Accuracy']
            
            st.dataframe(
                display_perf.style.format({
                    'Top-1 Accuracy': '{:.1%}', 
                    'Top-K Accuracy': '{:.1%}'
                }).background_gradient(subset=['Top-1 Accuracy', 'Top-K Accuracy'], cmap='RdYlGn'),
                hide_index=True, 
                width="stretch"
            )
        else:
            st.info("No detailed CSV file found. Run `--test classify` to generate full breakdown statistics.")
    else:
        st.warning("No classification JSON report found.")

# ==========================================
# TAB 2: KB COVERAGE
# ==========================================
with tab2:
    st.header("KB Coverage (Module 2)")
    if kb_data:
        st.info(f"Loaded: `{os.path.basename(kb_file)}`")
        col1, col2, col3 = st.columns(3)
        total = kb_data.get('total_conditions', 0)
        covered = kb_data.get('covered_count', 0)
        missing = total - covered
        
        col1.metric(
            "Total Diagnoses (conditions.txt)", 
            total,
            help="Total number of supported diagnosis classes in the application."
        )
        col2.metric(
            "Covered in Knowledge Base", 
            covered,
            help="Number of conditions that have a dedicated, detailed response template in clinical_kb.json."
        )
        
        col3.metric(
            "Coverage Rate", 
            f"{kb_data.get('coverage_pct', 0):.1f}%",
            help="Percentage of the full vocabulary that has a dedicated KB entry."
        )
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Overview")
            st.write(f"The DermLIP system can predict **{total}** unique conditions.")
            st.write(f"Of those, **{covered}** ({kb_data.get('coverage_pct', 0):.1f}%) have a highly detailed entry in the `clinical_kb.json` file. This means if the model predicts one of these, it can pull specific symptoms, causes, and treatments from the database.")
            st.write(f"The remaining **{missing}** conditions are *missing* from the Knowledge Base. If the model predicts these, it will fall back to a generic response template.")
            
            fig_pie = px.pie(
                values=[covered, missing], 
                names=['Covered (Has KB Entry)', 'Missing Generic Fallback'], 
                hole=0.4,
                title="KB Coverage Ratio", 
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial')
            st.plotly_chart(fig_pie, width="stretch")
            
        with c2:
            st.subheader("Missing Conditions")
            st.write("These conditions are in `conditions.txt` but lack a matching entry in `clinical_kb.json`.")
            missing_list = kb_data.get('uncovered', []) if 'uncovered' in kb_data else []
            # handle 'uncovered_count' fallback if the actual list isn't there
            if not missing_list and kb_data.get('uncovered_count', 0) > 0:
                missing_list = ["(See full report for uncovered conditions list)"]
                
            if missing_list:
                st.dataframe(pd.DataFrame(missing_list, columns=["Condition Vocabulary Key"]), hide_index=True, width="stretch")
            else:
                st.success("100% Coverage! No missing entries.")
    else:
        st.warning("No KB Coverage report found.")

# ==========================================
# TAB 3: LATENCY
# ==========================================
with tab3:
    st.header("Latency Benchmark (Module 3)")
    lat_data = None
    # Try full_eval first for latency, fallback to dedicated latency if it exists
    if full_eval_data and "latency" in full_eval_data:
        lat_data = full_eval_data["latency"]
    
    if lat_data:
        st.info(f"Loaded from: `{os.path.basename(latency_file)}`")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Throughput", f"{lat_data.get('requests_per_sec', 0):.1f} req/sec")
        col2.metric("Median (p50)", f"{lat_data.get('classify_p50_ms', 0):.0f} ms")
        col3.metric("95th Percentile (p95)", f"{lat_data.get('classify_p95_ms', 0):.0f} ms")
        col4.metric("99th Percentile (p99)", f"{lat_data.get('classify_p99_ms', 0):.0f} ms")
        
        st.markdown("---")
        st.subheader("Chat Response Latency (per question)")
        chat_lats = lat_data.get("chat_latency_ms", {})
        if chat_lats:
            df_lat = pd.DataFrame(list(chat_lats.items()), columns=['Question', 'Latency (ms)'])
            fig2 = px.bar(
                df_lat, 
                x='Latency (ms)', 
                y='Question', 
                orientation='h', 
                title="Latency by Follow-up Prompt", 
                color='Latency (ms)', 
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("No chat latency data available in this run.")
    else:
        st.warning("No Latency benchmark report found in `full_eval_*.json`.")

# ==========================================
# TAB 4: CHAT QUALITY
# ==========================================
with tab4:
    st.header("Chat Quality (Module 4)")
    if chat_data:
        st.info(f"Loaded: `{os.path.basename(chat_file)}`")
        
        total = 0
        loop_rate = 0
        trunc_rate = 0
        avg_len = 0
        
        if full_eval_data and 'chat' in full_eval_data:
            c = full_eval_data['chat']
            total = c.get('total_questions', 0)
            loop_rate = (c.get('loop_count', 0) / total * 100) if total else 0
            trunc_rate = (c.get('truncated_count', 0) / total * 100) if total else 0
            avg_len = c.get('avg_reply_length', 0)
        else:
            total = len(chat_data)
            if total > 0:
                loops = sum(1 for r in chat_data if r.get('looping'))
                truncs = sum(1 for r in chat_data if r.get('truncated'))
                lengths = [r.get('length', 0) for r in chat_data]
                loop_rate = (loops / total) * 100
                trunc_rate = (truncs / total) * 100
                avg_len = sum(lengths) / total

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Prompts", total)
        col2.metric("Loop / Stutter Rate", f"{loop_rate:.1f}%")
        col3.metric("Truncation Rate", f"{trunc_rate:.1f}%")
        col4.metric("Avg Response Length", f"{avg_len:.0f} chars")
        
        st.markdown("---")
        st.subheader("Conversation Transcript")
        
        for idx, q in enumerate(chat_data):
            with st.container():
                question_text = q.get('question', 'Unknown Question')
                st.markdown(f"**🗣️ Patient:** {question_text}")
                
                # Check metrics for alerting
                is_err = q.get('looping') or q.get('truncated')
                box_color = "error" if is_err else "info"
                
                reply_text = q.get('reply', '(Full reply text not saved in report, only metadata)')
                
                if box_color == "error":
                    st.error(f"**🤖 AI:** {reply_text}")
                else:
                    st.info(f"**🤖 AI:** {reply_text}")
                
                tags = []
                tags.append(f"⏱️ {q.get('latency_ms', 0)} ms")
                tags.append(f"📏 {q.get('reply_length', 0)} chars")
                if q.get('looping'): tags.append("🚨 **LOOPING DETECTED**")
                if q.get('truncated'): tags.append("✂️ **TRUNCATED**")
                
                st.caption(" | ".join(tags))
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.warning("No Chat Quality reports found.")

# ==========================================
# TAB 5: LLM POLISH
# ==========================================
with tab5:
    st.header("LLM Polish Reliability (Module 5)")
    if llm_data:
        st.info(f"Loaded: `{os.path.basename(llm_file)}`")
        
        total = 0
        acc_rate = 0
        avg_lat = 0
        
        if full_eval_data and 'llm' in full_eval_data:
            l = full_eval_data['llm']
            total = l.get('total_tested', 0)
            acc_rate = l.get('acceptance_rate_pct', 0)
            avg_lat = l.get('avg_latency_ms', 0)
        else:
            total = len(llm_data)
            if total > 0:
                accs = sum(1 for r in llm_data if r.get('accepted'))
                lats = [r.get('latency_ms', 0) for r in llm_data]
                acc_rate = (accs / total) * 100
                avg_lat = sum(lats) / total

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tested Templates", total)
        col2.metric("Acceptance Rate", f"{acc_rate:.1f}%")
        col3.metric("Avg Latency", f"{avg_lat:.0f} ms")
        
        st.markdown("---")
        st.subheader("Rewrite Results")
        
        per_condition = llm_data
        if per_condition:
            df_llm = pd.DataFrame(per_condition)
            
            # Aggregate to get counts
            grouped_llm = df_llm.groupby('condition').agg(
                Total_Templates=('condition', 'count'),
                Accepted_Count=('accepted', 'sum'),
                Avg_Latency=('latency_ms', 'mean'),
                Avg_Raw_Length=('raw_length', 'mean'),
                Avg_Polished_Length=('polished_length', 'mean')
            ).reset_index()
            
            grouped_llm['Total_Failed'] = grouped_llm['Total_Templates'] - grouped_llm['Accepted_Count']
            
            # Format the dataframe for display
            display_df = grouped_llm[['condition', 'Total_Templates', 'Accepted_Count', 'Total_Failed', 'Avg_Latency']].copy()
            
            # Round latency to clean up the table
            display_df['Avg_Latency'] = display_df['Avg_Latency'].round(0).astype(int)
            display_df.columns = ['Condition', 'Total Tested', 'Total Accepted', 'Total Failed', 'Avg Latency (ms)']
            
            # Apply styling
            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch"
            )
            
            # Show a chart comparing character lengths
            st.markdown("---")
            st.subheader("Raw vs Polished Character Length")
            fig = px.bar(
                grouped_llm,
                x='condition',
                y=['Avg_Raw_Length', 'Avg_Polished_Length'],
                title="Average Template Length Expansion",
                barmode='group',
                labels={'value': 'Avg Character Count', 'condition': 'Condition', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("No LLM Polish report found. Run `python evaluate.py --test llm` to generate one.")
