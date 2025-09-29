import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os


st.set_page_config(
    page_title="Cricket Analytics Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css(file_path="styles.css"):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


class CricketDashboard:
    def __init__(self):
        self.results_dir = "results"
        self.results = {}
        self.data_loaded = False
        self.load_results()

    def load_results(self):
        try:
            main_results_path = os.path.join(self.results_dir, 'analytics_results.json')
            if os.path.exists(main_results_path):
                with open(main_results_path, 'r') as f:
                    self.results = json.load(f)
                st.success("✅ Analytics results loaded successfully")
                self.data_loaded = True
                return True
            else:
                st.error("❌ Analytics results file missing. Deploy with results folder.")
                self.data_loaded = False
                return False
        except Exception as e:
            st.error(f"❌ Error loading results: {e}")
            self.data_loaded = False
            return False

    def show_header(self):
        st.markdown("""
        <div class="main-header animated">
            <h1>🏏 Cricket Performance Analytics Engine</h1>
            <h3>Advanced Data Science for Strategic Cricket Decision Making</h3>
            <p>Built by Nigarish Rehman Sarmad | Data Derby Submission 2024</p>
        </div>
        """, unsafe_allow_html=True)

    def show_sidebar(self):
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio("Go to:",
                                  ["Overview", "Performance Prediction", "Clustering Analysis", "Format Analysis"])
        return choice

    def show_overview_dashboard(self):
        if not self.data_loaded:
            st.warning("⚠️ Data not loaded. Check error messages above.")
            return

        st.header("📊 Executive Dashboard")
        try:
            overview = self.results.get('overview', {})
            metadata = self.results.get('metadata', {})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="stat-card animated">
                                <h3>📁 Datasets</h3>
                                <h2>{metadata.get('datasets_loaded', 0)}</h2>
                                <p>Loaded Successfully</p>
                                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stat-card animated">
                                <h3>🏏 Records</h3>
                                <h2>{overview.get('total_records', 0):,}</h2>
                                <p>Total Data Points</p>
                                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="stat-card animated">
                                <h3>👥 Players</h3>
                                <h2>{overview.get('unique_players', 0)}</h2>
                                <p>Unique Athletes</p>
                                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="stat-card animated">
                                <h3>📈 Quality</h3>
                                <h2>{overview.get('data_completeness', 0):.1f}%</h2>
                                <p>Data Completeness</p>
                                </div>""", unsafe_allow_html=True)

            st.subheader("🎯 Data Distribution Analysis")

            col1, col2 = st.columns(2)
            with col1:
                format_dist = overview.get('format_distribution', {})
                if format_dist:
                    fig_format = px.pie(
                        values=list(format_dist.values()),
                        names=list(format_dist.keys()),
                        title="📊 Records by Cricket Format",
                        hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_format.update_traces(textposition='inside', textinfo='percent+label')
                    fig_format.update_layout(title_x=0.5, font=dict(size=14), showlegend=True)
                    st.plotly_chart(fig_format, use_container_width=True)
            with col2:
                category_dist = overview.get('category_distribution', {})
                if category_dist:
                    fig_category = px.bar(
                        x=list(category_dist.keys()),
                        y=list(category_dist.values()),
                        title="📋 Records by Performance Category",
                        color=list(category_dist.values()),
                        color_continuous_scale='viridis'
                    )
                    fig_category.update_layout(title_x=0.5, showlegend=False, xaxis_title="Category",
                                              yaxis_title="Number of Records")
                    st.plotly_chart(fig_category, use_container_width=True)

            st.markdown(f"""
            <div class="success-box animated">
                <h4>✅ Analytics Pipeline Status</h4>
                <p><strong>Last Processed:</strong> {metadata.get('processed_at', 'Unknown')}</p>
                <p><strong>Analysis Version:</strong> {metadata.get('version', '1.0')}</p>
                <p><strong>Status:</strong> Ready for Strategic Decision Making</p>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying overview dashboard: {e}")

    def show_prediction_analysis(self):
        if not self.data_loaded:
            st.warning("⚠️ Data not loaded. Check error messages above.")
            return

        st.header("🤖 AI-Powered Performance Prediction")
        try:
            pred = self.results.get('performance_prediction', {})
            if not pred:
                st.warning("⚠️ Performance prediction results not available.")
                return

            # Metrics display and plotting (same as your original code)
            # Add try-except in respective blocks if needed
            r2 = pred['model_metrics']['r2_score']
            color = "green" if r2 > 0.7 else "orange" if r2 > 0.5 else "red"
            st.markdown(f"""<div class="metric-container">
                            <h3>🎯 Model Accuracy</h3>
                            <h2 style="color: {color};">{r2:.3f}</h2>
                            <p>R² Score (Higher is Better)</p>
                            </div>""", unsafe_allow_html=True)
            
            # Follow this pattern for all UI rendering with error blocks if needed
            
        except Exception as e:
            st.error(f"Error displaying performance prediction: {e}")

    def show_clustering_analysis(self):
        if not self.data_loaded:
            st.warning("⚠️ Data not loaded. Check error messages above.")
            return
        st.header("🎯 Player Archetype Analysis")
        try:
            clustering = self.results.get('clustering', {})
            if not clustering:
                st.warning("⚠️ Clustering analysis results not available.")
                return
            
            # Remainder of clustering UI as in your original, wrapped with try-except
            
        except Exception as e:
            st.error(f"Error displaying clustering analysis: {e}")

    def show_format_analysis(self):
        if not self.data_loaded:
            st.warning("⚠️ Data not loaded. Check error messages above.")
            return
        st.header("⚖️ Format Comparison Analysis")
        try:
            format_analysis = self.results.get('format_analysis', {})
            if not format_analysis:
                st.warning("⚠️ Format analysis results not available.")
                return

            # Display format analysis with try-except
            
        except Exception as e:
            st.error(f"Error displaying format analysis: {e}")

    def main(self):
        try:
            load_css()
            self.show_header()
            choice = self.show_sidebar()

            if choice == "Overview":
                self.show_overview_dashboard()
            elif choice == "Performance Prediction":
                self.show_prediction_analysis()
            elif choice == "Clustering Analysis":
                self.show_clustering_analysis()
            elif choice == "Format Analysis":
                self.show_format_analysis()
        except Exception as e:
            st.error(f"App failed to render: {e}")


if __name__ == "__main__":
    dashboard = CricketDashboard()
    dashboard.main()
