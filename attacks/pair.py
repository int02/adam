class PAIRAttack:
    def __init__(self, llm):
        self.llm = llm
    async def run(self, base_prompt):
        prompt = f"[PAIR] {base_prompt}"
        return await self.llm.generate(prompt)
