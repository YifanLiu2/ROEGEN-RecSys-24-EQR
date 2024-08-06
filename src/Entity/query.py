class AbstractQuery:
    """
    Represents a user's travel destinations query.
    """
    def __init__(self, description: str):
        self.description = description
        self.reformulation = None

    def get_description(self) -> str:
        return self.description

    def set_reformulation(self, reformulation: str):
        self.reformulation = reformulation

    def get_reformulation(self) -> str:
        if self.reformulation:
            return self.reformulation
        else:
            return self.description