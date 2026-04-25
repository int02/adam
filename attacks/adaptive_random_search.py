class AdaptiveRandomSearchAttack:
    def __init__(self, llm):
        self.llm = llm
    async def run(self, base_prompt):
        # Simple stub for adaptive random search attack
        prompt = f"[ARS] {base_prompt}"
        return await self.llm.generate(prompt)
