import streamlit as st


class Sidebar:
    """
    Sidebar
    """

    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.0
    TEMPERATURE_STEP = 0.01

    @staticmethod
    def about():
        about = st.sidebar.expander("About 🤖")
        sections = [
            "#### mAI CFO is a digital AI CFO, that powers: TODO ",
        ]
        for section in sections:
            about.write(section)

    @staticmethod
    def integrate_bank_account_data():
        if st.button("Integrate Bank Account data"):
            st.session_state["integrate_bank_data"] = True
        st.session_state.setdefault("integrate_bank_data", False)

    @staticmethod
    def reset_chat_button():
        if st.button("Reset chat"):
            st.session_state["reset_chat"] = True
        st.session_state.setdefault("reset_chat", False)

    def model_selector(self):
        model = st.selectbox(label="Model", options=self.MODEL_OPTIONS)
        st.session_state["model"] = model

    def temperature_slider(self):
        temperature = st.slider(
            label="Temperature",
            min_value=self.TEMPERATURE_MIN_VALUE,
            max_value=self.TEMPERATURE_MAX_VALUE,
            value=self.TEMPERATURE_DEFAULT_VALUE,
            step=self.TEMPERATURE_STEP,
        )
        st.session_state["temperature"] = temperature

    def show_options(self):
        with st.sidebar.expander("Tools and Integrations", expanded=False):
            self.reset_chat_button()
            self.integrate_bank_account_data()
            # self.model_selector()
            # self.temperature_slider()
            st.session_state.setdefault("model", self.MODEL_OPTIONS[0])
            st.session_state.setdefault("temperature", self.TEMPERATURE_DEFAULT_VALUE)
