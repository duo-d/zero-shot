import openai
from openai import OpenAI
from config import OPENAI_API_KEY, SIM_PARAMS
from llm_client import LLMClient

class AgentAction:
    """
    Action Module (§4.1.4.2)
    Task1~Task4: Simulate and invoke Memory's retrieval, writing, and reflection operations.
    """
    def __init__(self, profile, memory):
        self.profile = profile
        self.memory = memory
        self.llm = LLMClient()


    def _call_llm(self, messages):
        print (messages)
        return self.llm.call(messages)

    def simulate_step(self, practice, time_step, similarity_fn):
        # 1. Memory Retrieval
        short_mem = self.memory.retrieve_short()
        long_mem = self.memory.retrieve_long()

        # 2. Build prompt: include profile + short-term + long-term memories
        prompt = self._build_prompt(practice, short_mem, long_mem)
        messages = [
            {'role': 'system', 'content': self.profile.build_prompt()},
            {'role': 'user', 'content': prompt}
        ]

        # 3. LLM Generated Task1~Task4
        resp = self._call_llm(messages)

        # 4. Analysis Results
        ans = {}
        for i in range(1, 5):
            tag = f'Task{i}:'
            if tag in resp:
                ans[f'task{i}'] = resp.split(tag)[1].split('\n')[0].strip()

        # 5. Memory Writing: write factual
        record = [
            practice['exer_content'],
            practice['know_name'],
            practice['score'],
            1
        ]
        self.memory.write_factual(record)

        # 6. Memory Reinforcement & Long-term update
        # sims = similarity_fn(record)
        if SIM_PARAMS['learning_effect'] == 'yes':
            self.memory.reinforce(record)

        # 7. Forgetting
        if SIM_PARAMS['forgetting_effect'] == 'yes':
            self.memory.forget(time_step)

        # 8. Update Short-term
        self.memory.retrieve_short()

        # 9. Memory Reflection
        # 9a. Corrective Reflection
        corrective = self.memory.reflect_corrective(practice, ans)
        # 9b. Summary Reflection
        reflect = self.memory.reflect_summary(corrective)
        
        messages.append({"role": "assistant",
                         "content": resp})
        messages.append({"role": "user",
                         "content": reflect})
        summary = self._call_llm(messages)
        
        self.memory.write_long_summary(summary)
        self.memory.write_know(practice)

        # 10. Return
        return ans, resp, corrective, summary

    def _build_prompt(self, practice, short_mem, long_mem):
        """
        Construct a prompt that meets the requirements of Tasks 1–4 in the paper, including the following in sequence:
        1. Profile Prompt
        2. Short-term Memory Search results
        3. Long-term Memory Search results (Enhancing factual knowledge, mastery of knowledge, and learning state)
        4. Practice Content and Four Task Descriptions
        """
        prompt = (
            f"Recommended Exercise:\n"
            f"- Textual Content: {practice['exer_content']}\n"
            f"- Knowledge Concept (true): {practice['know_name']}\n"
        )

        # Task1: Cognition-driven Action Prompt
        prompt += (
            "\nTask 1: Based on your Profile and Knowledge Proficiency, "
            "decide whether you want to attempt this exercise. "
            "If too difficult, output 'No'; otherwise, output 'Yes'.\n"
        )

        # Short-Term Memory Demonstration
        if short_mem:
            prompt += "\nYour Short-term Memory (recent facts):\n"
            for idx, r in enumerate(short_mem, 1):
                prompt += (
                    f" Record {idx}: Content='{r[0]}', Concept={r[1]}, Correct={r[2]}\n"
                )
            prompt += "\n"

        # Task2: Concept Recognition Prompt
        prompt += (
            "Task 2: Identify the knowledge concept tested by this exercise. "
            "Choose one from the following options:\n"
        )
        options = [practice['know_name']] + long_mem.get('practiced_knowledge', [])[:2]
        for opt in options:
            prompt += f" - {opt}\n"
        prompt += "Only output the concept name.\n"

        # Strengthening facts and learning states
        if long_mem.get('significant_facts'):
            prompt += "\nYour Long-term Memory (reinforced facts):\n"
            for idx, f in enumerate(long_mem['significant_facts'], 1):
                prompt += f" Record {idx}: Concept={f[1]}, ReinforcedTimes={f[3]}\n"
        if long_mem.get('learning_status'):
            prompt += (
                f"\nYour current Learning Status Summary: "
                f"{long_mem['learning_status'][-1]}\n"
            )

        # Task3: Problem-Solving Approach and Answer
        prompt += (
            "\nTask 3: Propose a concise problem-solving idea based on your Profile and Memories, "
            "then give a final answer.\n"
        )
        
        # Task4: Accuracy Prediction
        prompt += (
            "\nTask 4: Predict whether you will answer correctly ('Yes' or 'No') based on the idea.\n"
        )

        # Output Format Specifications
        prompt += (
            "\nOutput format exactly as:\n"
            "Task1: <Yes/No>\n"
            "Task2: <concept>\n"
            "Task3: <your idea and final answer>\n"
            "Task4: <Yes/No>\n"
        )
        return prompt
