class ImprompterToolCall:
    def __init__(self, llm):
        self.llm = llm
    async def run(self, base_prompt):
        prompt = f"[ImprompterToolCall] {base_prompt}"
        return await self.llm.generate(prompt)
