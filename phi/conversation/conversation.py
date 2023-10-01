import json
from uuid import uuid4
from datetime import datetime
from typing import List, Any, Optional, Dict, Iterator, Callable, cast, Union

from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field

from phi.conversation.memory import ConversationMemory
from phi.conversation.schemas import ConversationRow
from phi.conversation.storage import ConversationStorage
from phi.document import Document
from phi.knowledge.base import KnowledgeBase
from phi.llm.base import LLM
from phi.llm.openai import OpenAIChat
from phi.llm.schemas import Message, References
from phi.llm.function.registry import FunctionRegistry
from phi.utils.format_str import remove_indent
from phi.utils.log import logger, set_log_level_to_debug
from phi.utils.timer import Timer


class Conversation(BaseModel):
    # -*- LLM to use for this conversation
    llm: LLM = OpenAIChat()
    # -*- LLM Introduction
    # Add an introduction (from the LLM) to the chat history
    introduction: Optional[str] = None

    # -*- User settings
    # Name and type of user participating in this conversation.
    user_name: Optional[str] = None
    user_type: Optional[str] = None

    # -*- Conversation settings
    # Conversation UUID
    id: Optional[str] = Field(None, validate_default=True)
    # Conversation name
    name: Optional[str] = None
    # True if this conversation is active i.e. not ended
    is_active: bool = True
    # Metadata associated with this conversation
    meta_data: Optional[Dict[str, Any]] = None
    # Extra data associated with this conversation
    extra_data: Optional[Dict[str, Any]] = None
    # The timestamp of when this conversation was created in the database
    created_at: Optional[datetime] = None
    # The timestamp of when this conversation was last updated in the database
    updated_at: Optional[datetime] = None

    # Monitor conversations on phidata.com
    monitoring: bool = False

    # -*- Conversation Memory
    memory: ConversationMemory = ConversationMemory()
    # Add chat history to the prompt sent to the LLM.
    # If True, a formatted chat history is added to the default user_prompt.
    add_chat_history_to_prompt: bool = False
    # Add chat history to the messages sent to the LLM.
    # If True, the chat history is added to the messages sent to the LLM.
    add_chat_history_to_messages: bool = False
    # Number of previous messages to add to prompt or messages sent to the LLM.
    num_history_messages: int = 8

    # -*- Conversation Knowledge Base
    knowledge_base: Optional[KnowledgeBase] = None
    # Add references from the knowledge base to the prompt sent to the LLM.
    add_references_to_prompt: bool = False

    # -*- Conversation Storage
    storage: Optional[ConversationStorage] = None
    # Create table if it doesn't exist
    create_storage: bool = True
    # ConversationRow from the database: DO NOT SET THIS MANUALLY
    database_row: Optional[ConversationRow] = None

    # -*- Enable Function Calls
    # Makes the conversation autonomous.
    function_calls: bool = False
    # Add a list of default functions to the LLM
    default_functions: bool = True
    # Show function calls in LLM messages.
    show_function_calls: bool = False
    # A list of functions to add to the LLM.
    functions: Optional[List[Callable]] = None
    # A list of function registries to add to the LLM.
    function_registries: Optional[List[FunctionRegistry]] = None

    #
    # -*- Prompt Settings
    #
    # -*- System prompt: provide the system prompt as a string or using a function
    system_prompt: Optional[str] = None
    # Function to build the system prompt.
    # This function is provided the conversation as an argument
    #   and should return the system_prompt as a string.
    # Signature:
    # def system_prompt_function(conversation: Conversation) -> str:
    #    ...
    system_prompt_function: Optional[Callable[..., Optional[str]]] = None
    # -*- User prompt: provide the user prompt as a string or using a function
    user_prompt: Optional[str] = None
    # Function to build the user prompt.
    # This function is provided the conversation and the user message as arguments
    #   and should return the user_prompt as a string.
    # If add_references_to_prompt is True, then references are also provided as an argument.
    # If add_chat_history_to_prompt is True, then chat_history is also provided as an argument.
    # Signature:
    # def custom_user_prompt_function(
    #     conversation: Conversation,
    #     message: str,
    #     references: Optional[str] = None,
    #     chat_history: Optional[str] = None,
    # ) -> str:
    #     ...
    user_prompt_function: Optional[Callable[..., str]] = None

    # -*- Functions to customize the user_prompt
    # Function to build references for the default user_prompt
    # This function, if provided, is called when add_references_to_prompt is True
    # Signature:
    # def references(conversation: Conversation, query: str) -> Optional[str]:
    #     ...
    references_function: Optional[Callable[..., Optional[str]]] = None
    # Function to build the chat_history for the default user prompt
    # This function, if provided, is called when add_chat_history_to_prompt is True
    # Signature:
    # def chat_history(conversation: Conversation) -> str:
    #     ...
    chat_history_function: Optional[Callable[..., Optional[str]]] = None

    # If True, show debug logs
    debug_mode: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("id", mode="before")
    def set_conversation_id(cls, v: Optional[str]) -> str:
        return v if v is not None else str(uuid4())

    @field_validator("debug_mode", mode="before")
    def set_log_level(cls, v: bool) -> bool:
        if v:
            set_log_level_to_debug()
            logger.debug("Debug logs enabled")
        return v

    @model_validator(mode="after")
    def add_functions_to_llm(self) -> "Conversation":
        if self.function_calls:
            if self.default_functions:
                default_func_list: List[Callable] = [
                    self.get_last_n_chats,
                    self.search_knowledge_base,
                ]
                for func in default_func_list:
                    self.llm.add_function(func)

            # Add functions from self.functions
            if self.functions is not None:
                for func in self.functions:
                    self.llm.add_function(func)

            # Add functions from registries
            if self.function_registries is not None:
                for registry in self.function_registries:
                    self.llm.add_function_registry(registry)

            # Set function call to auto if it is not set
            if self.llm.function_call is None:
                self.llm.function_call = "auto"

            # Set show_function_calls if it is not set on the llm
            if self.llm.show_function_calls is None:
                self.llm.show_function_calls = self.show_function_calls
        return self

    def to_database_row(self) -> ConversationRow:
        """Create a ConversationRow for the current conversation (to save to the database)"""

        return ConversationRow(
            id=self.id,
            name=self.name,
            user_name=self.user_name,
            user_type=self.user_type,
            is_active=self.is_active,
            llm=self.llm.to_dict(),
            memory=self.memory.to_dict(),
            meta_data=self.meta_data,
            extra_data=self.extra_data,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def from_database_row(self, row: ConversationRow):
        """Load the existing conversation from a ConversationRow (from the database)"""

        # Values that are overwritten from the database if they are not set in the conversation
        if self.id is None and row.id is not None:
            self.id = row.id
        if self.name is None and row.name is not None:
            self.name = row.name
        if self.user_name is None and row.user_name is not None:
            self.user_name = row.user_name
        if self.user_type is None and row.user_type is not None:
            self.user_type = row.user_type
        if self.is_active is None and row.is_active is not None:
            self.is_active = row.is_active

        # Update llm data from the ConversationRow
        if row.llm is not None:
            # Update llm metrics from the database
            llm_metrics_from_db = row.llm.get("metrics")
            if llm_metrics_from_db is not None and isinstance(llm_metrics_from_db, dict):
                try:
                    self.llm.metrics = llm_metrics_from_db
                except Exception as e:
                    logger.warning(f"Failed to load llm metrics: {e}")

            # # Update llm functions from the database
            # llm_functions_from_db = row.llm.get("functions")
            # if llm_functions_from_db is not None and isinstance(llm_functions_from_db, dict):
            #     try:
            #         for k, v in llm_functions_from_db.items():
            #             _llm_function = Function(**v)
            #             self.llm.add_function_schema(func=_llm_function, if_not_exists=True)
            #     except Exception as e:
            #         logger.error(f"Failed to load llm functions: {e}")

        # Update conversation memory from the ConversationRow
        if row.memory is not None:
            try:
                self.memory = self.memory.__class__.model_validate(row.memory)
            except Exception as e:
                logger.warning(f"Failed to load conversation memory: {e}")

        # Update meta_data from the database
        if row.meta_data is not None:
            # If meta_data is set in the conversation,
            # merge it with the database meta_data. The conversation meta_data takes precedence
            if self.meta_data is not None and row.meta_data is not None:
                self.meta_data = {**row.meta_data, **self.meta_data}
            # If meta_data is not set in the conversation, use the database meta_data
            if self.meta_data is None and row.meta_data is not None:
                self.meta_data = row.meta_data

        # Update extra_data from the database
        if row.extra_data is not None:
            # If extra_data is set in the conversation,
            # merge it with the database extra_data. The conversation extra_data takes precedence
            if self.extra_data is not None and row.extra_data is not None:
                self.extra_data = {**row.extra_data, **self.extra_data}
            # If extra_data is not set in the conversation, use the database extra_data
            if self.extra_data is None and row.extra_data is not None:
                self.extra_data = row.extra_data

        # Update the timestamp of when this conversation was created in the database
        if row.created_at is not None:
            self.created_at = row.created_at

        # Update the timestamp of when this conversation was last updated in the database
        if row.updated_at is not None:
            self.updated_at = row.updated_at

    def read_from_storage(self) -> Optional[ConversationRow]:
        """Load the conversation from storage"""

        if self.storage is not None and self.id is not None:
            self.database_row = self.storage.read(conversation_id=self.id)
            if self.database_row is not None:
                logger.debug(f"-*- Loading conversation: {self.database_row.id}")
                self.from_database_row(row=self.database_row)
                logger.debug(f"-*- Loaded conversation: {self.id}")
        return self.database_row

    def write_to_storage(self) -> Optional[ConversationRow]:
        """Save the conversation to the storage"""

        if self.storage is not None:
            if self.create_storage:
                self.storage.create()
            self.database_row = self.storage.upsert(conversation=self.to_database_row())
        return self.database_row

    def add_introduction(self, introduction: str) -> None:
        """Add the introduction to the chat history"""

        if introduction is not None:
            if len(self.memory.chat_history) == 0:
                self.memory.add_chat_message(Message(role="assistant", content=introduction))

    def start(self) -> Optional[str]:
        """Start the conversation and return the conversation ID. This function:
        - Creates a new conversation in the storage if it does not exist
        - Load the conversation from the storage if it exists
        """

        # If a database_row exists, return the conversation_id
        if self.database_row is not None:
            return self.database_row.id

        # Create a new conversation or load an existing conversation
        if self.storage is not None:
            # Load existing conversation if it exists
            logger.debug(f"Reading conversation: {self.id}")
            self.read_from_storage()

            # Create a new conversation
            if self.database_row is None:
                logger.debug("-*- Creating new conversation")
                if self.introduction:
                    self.add_introduction(self.introduction)
                self.database_row = self.write_to_storage()
                if self.database_row is None:
                    raise Exception("Failed to create new conversation in storage")
                logger.debug(f"-*- Created conversation: {self.database_row.id}")
                self.from_database_row(row=self.database_row)
                self._api_upsert_conversation()
        return self.id

    def end(self) -> None:
        """End the conversation"""
        if self.storage is not None and self.id is not None:
            self.storage.end(conversation_id=self.id)
        self.is_active = False

    def get_system_prompt(self) -> str:
        """Return the system prompt for the conversation"""

        if self.system_prompt is not None:
            return "\n".join([line.strip() for line in self.system_prompt.split("\n")])

        if self.system_prompt_function is not None:
            system_prompt_kwargs = {"conversation": self}
            _system_prompt_from_function = remove_indent(self.system_prompt_function(**system_prompt_kwargs))
            if _system_prompt_from_function is not None:
                return _system_prompt_from_function
            else:
                raise Exception("system_prompt_function returned None")

        _system_prompt = "You are a chatbot "

        if self.user_type:
            _system_prompt += f"that is designed to help a '{self.user_type}' with their work.\n"
        else:
            _system_prompt += "that is designed to help a user with their work.\n"

        if self.knowledge_base is not None:
            _system_prompt += "You have access to a knowledge base that you can search for information.\n"
            _system_prompt += (
                "If you cannot find the answer in the knowledge base, "
                "say that you cannot find the answer in the knowledge base.\n"
            )

        if self.function_calls:
            _system_prompt += "You have access to functions that you can run to help you respond to the user.\n"

        _system_prompt += """Remember the following guidelines:
        - If you don't know the answer, say 'I don't know'.
        - Use bullet points where possible.
        - User markdown to format your answers.
        """

        # Return the system prompt after removing newlines and indenting
        _system_prompt = cast(str, remove_indent(_system_prompt))
        return _system_prompt

    def get_references_from_knowledge_base(self, query: str, num_documents: Optional[int] = None) -> Optional[str]:
        """Return a list of references from the knowledge base"""

        if self.references_function is not None:
            reference_kwargs = {"conversation": self, "query": query}
            return remove_indent(self.references_function(**reference_kwargs))

        if self.knowledge_base is None:
            return None

        relevant_docs: List[Document] = self.knowledge_base.search(query=query, num_documents=num_documents)
        return json.dumps([doc.to_dict() for doc in relevant_docs])

    def get_formatted_chat_history(self) -> Optional[str]:
        """Returns a formatted chat history to use in the user prompt"""

        if self.chat_history_function is not None:
            chat_history_kwargs = {"conversation": self}
            return remove_indent(self.chat_history_function(**chat_history_kwargs))

        formatted_history = self.memory.get_formatted_chat_history(num_messages=self.num_history_messages)
        if formatted_history == "":
            return None
        return remove_indent(formatted_history)

    def get_user_prompt(
        self, message: str, references: Optional[str] = None, chat_history: Optional[str] = None
    ) -> str:
        """Build the user prompt given a message, references and chat_history"""

        if self.user_prompt_function is not None:
            user_prompt_kwargs = {
                "conversation": self,
                "message": message,
                "references": references,
                "chat_history": chat_history,
            }
            _user_prompt_from_function = remove_indent(self.user_prompt_function(**user_prompt_kwargs))
            if _user_prompt_from_function is not None:
                return _user_prompt_from_function
            else:
                raise Exception("user_prompt_function returned None")

        _user_prompt = "Your task is to respond to the following message"
        if self.user_type:
            _user_prompt += f" from a '{self.user_type}'"
        _user_prompt += " in the best way possible.\n"

        # Add references to prompt
        if references:
            _user_prompt += f"""\n
                You can use the following information from the knowledge base if it helps respond to the message.
                START OF KNOWLEDGE BASE INFORMATION
                ```
                {references}
                ```
                END OF KNOWLEDGE BASE INFORMATION
                """

        # Add chat_history to prompt
        if chat_history:
            _user_prompt += f"""\n
                You can use the following chat history to reference past messages.
                START OF CHAT HISTORY
                ```
                {chat_history}
                ```
                END OF CHAT HISTORY
                """

        # Remind the LLM of its task
        if references or chat_history:
            _user_prompt += "\nRemember, your task is to respond to the following message in the best way possible."

        _user_prompt += f"\nUSER: {message}"
        _user_prompt += "\nASSISTANT: "

        # Return the user prompt after removing newlines and indenting
        _user_prompt = cast(str, remove_indent(_user_prompt))
        return _user_prompt

    def _chat(self, message: str, stream: bool = True) -> Iterator[str]:
        logger.debug("*********** Conversation Chat Start ***********")
        # Load the conversation from the database if available
        self.read_from_storage()

        # -*- Build the system prompt
        system_prompt = self.get_system_prompt()

        # -*- References to add to the user_prompt and send to the api for monitoring
        references: Optional[References] = None

        # -*- Get references to add to the user_prompt
        user_prompt_references = None
        if self.add_references_to_prompt:
            reference_timer = Timer()
            reference_timer.start()
            user_prompt_references = self.get_references_from_knowledge_base(query=message)
            reference_timer.stop()
            references = References(
                query=message, references=user_prompt_references, time=round(reference_timer.elapsed, 4)
            )
            logger.debug(f"Time to get references: {reference_timer.elapsed:.4f}s")

        # -*- Get chat history to add to the user prompt
        user_prompt_chat_history = None
        if self.add_chat_history_to_prompt:
            user_prompt_chat_history = self.get_formatted_chat_history()

        # -*- Build the user prompt
        user_prompt = self.get_user_prompt(
            message=message, references=user_prompt_references, chat_history=user_prompt_chat_history
        )

        # -*- Build the messages to send to the LLM
        # Create system message
        system_prompt_message = Message(role="system", content=system_prompt)
        # Create user message
        user_prompt_message = Message(role="user", content=user_prompt)

        # Create message list
        messages: List[Message] = [system_prompt_message]
        if self.add_chat_history_to_messages:
            messages += self.memory.get_last_n_messages(last_n=self.num_history_messages)
        messages += [user_prompt_message]

        # -*- Generate response (includes running function calls)
        llm_response = ""
        if stream:
            for response_chunk in self.llm.parsed_response_stream(messages=messages):
                llm_response += response_chunk
                yield response_chunk
        else:
            llm_response = self.llm.parsed_response(messages=messages)

        # -*- Add messages to the memory
        # Add the system prompt to the memory - added only if this is the first message to the LLM
        self.memory.add_system_prompt(message=system_prompt_message)

        # Add user message to the memory - this is added to the chat_history
        self.memory.add_user_message(message=Message(role="user", content=message))

        # Add user prompt to the memory - this is added to the llm_messages
        self.memory.add_llm_message(message=user_prompt_message)

        # Add references to the memory
        if references:
            self.memory.add_references(references=references)

        # Add llm response to the memory - this is added to the chat_history and llm_messages
        self.memory.add_llm_response(message=Message(role="assistant", content=llm_response))

        # -*- Save conversation to storage
        self.write_to_storage()

        # -*- Send conversation event for monitoring
        event_data = {
            "user_message": message,
            "llm_response": llm_response,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "references": references.model_dump(exclude_none=True) if references else None,
            "metrics": self.llm.metrics,
        }
        self._api_send_conversation_event(event_type="chat", event_data=event_data)

        # -*- Yield final response if not streaming
        if not stream:
            yield llm_response
        logger.debug("*********** Conversation Chat End ***********")

    def chat(self, message: str, stream: bool = True) -> Union[Iterator[str], str]:
        resp = self._chat(message=message, stream=stream)
        if stream:
            return resp
        else:
            return next(resp)

    def _chat_raw(
        self, messages: List[Message], user_message: Optional[str] = None, stream: bool = True
    ) -> Iterator[Dict]:
        logger.debug("*********** Conversation Raw Chat Start ***********")
        # Load the conversation from the database if available
        self.read_from_storage()

        # -*- Add user message to the memory - this is added to the chat_history
        if user_message:
            self.memory.add_user_message(Message(role="user", content=user_message))

        # -*- Add prompts to the memory - these are added to the llm_messages
        self.memory.add_llm_messages(messages=messages)

        # -*- Generate response
        batch_llm_response_message = {}
        if stream:
            for response_delta in self.llm.response_delta(messages=messages):
                yield response_delta
        else:
            batch_llm_response_message = self.llm.response_message(messages=messages)

        # Add llm response to the memory - this is added to the chat_history and llm_messages
        # LLM Response is the last message in the messages list
        llm_response_message = messages[-1]
        try:
            self.memory.add_llm_response(llm_response_message)
        except Exception as e:
            logger.warning(f"Failed to add llm response to memory: {e}")

        # -*- Save conversation to storage
        self.write_to_storage()

        # -*- Send conversation event for monitoring
        event_data = {
            "user_message": user_message,
            "llm_response": llm_response_message,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "metrics": self.llm.metrics,
        }
        self._api_send_conversation_event(event_type="chat_raw", event_data=event_data)

        # -*- Yield final response if not streaming
        if not stream:
            yield batch_llm_response_message
        logger.debug("*********** Conversation Raw Chat End ***********")

    def chat_raw(
        self, messages: List[Message], user_message: Optional[str] = None, stream: bool = True
    ) -> Union[Iterator[Dict], Dict]:
        resp = self._chat_raw(messages=messages, user_message=user_message, stream=stream)
        if stream:
            return resp
        else:
            return next(resp)

    def rename(self, name: str) -> None:
        """Rename the conversation"""
        self.read_from_storage()
        self.name = name

        # -*- Save conversation to storage
        self.write_to_storage()

        # -*- Update conversation
        self._api_upsert_conversation()

    def generate_name(self) -> str:
        """Generate a name for the conversation using chat history"""
        _conv = (
            "Please provide a suitable name for the following conversation in maximum 5 words.\n"
            "Remember, do not exceed 5 words.\n\n"
        )
        for message in self.memory.chat_history[1:6]:
            _conv += f"{message.role.upper()}: {message.content}\n"

        _conv += "\n\nConversation Name:"

        system_message = Message(
            role="system",
            content="Please provide a suitable name for the following conversation in maximum 5 words.",
        )
        user_message = Message(role="user", content=_conv)
        generate_name_messages = [system_message, user_message]
        generated_name = self.llm.parsed_response(messages=generate_name_messages)
        if len(generated_name.split()) > 15:
            logger.error("Generated name is too long. Trying again.")
            return self.generate_name()
        return generated_name.replace('"', "").strip()

    def auto_rename(self) -> None:
        """Automatically rename the conversation"""
        self.read_from_storage()
        generated_name = self.generate_name()
        logger.debug(f"Generated name: {generated_name}")
        self.name = generated_name

        # -*- Save conversation to storage
        self.write_to_storage()

        # -*- Update conversation
        self._api_upsert_conversation()

    ###########################################################################
    # Api functions
    ###########################################################################

    def _api_upsert_conversation(self):
        if not self.monitoring:
            return

        from phi.api.conversation import upsert_conversation, ConversationUpdate

        logger.debug("Sending conversation event")
        try:
            database_row: ConversationRow = self.database_row or self.to_database_row()
            upsert_conversation(
                conversation=ConversationUpdate(
                    conversation_key=database_row.conversation_key(),
                    conversation_data=database_row.conversation_data(),
                ),
            )
        except Exception as e:
            logger.debug(f"Could not log conversation event: {e}")

    def _api_send_conversation_event(
        self, event_type: str = "chat", event_data: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.monitoring:
            return

        from phi.api.conversation import log_conversation_event, ConversationEventCreate

        logger.debug("Sending conversation event")
        try:
            database_row: ConversationRow = self.database_row or self.to_database_row()
            log_conversation_event(
                conversation=ConversationEventCreate(
                    conversation_key=database_row.conversation_key(),
                    conversation_data=database_row.conversation_data(),
                    event_type=event_type,
                    event_data=event_data,
                ),
            )
        except Exception as e:
            logger.debug(f"Could not log conversation event: {e}")

    ###########################################################################
    # LLM functions
    ###########################################################################

    def get_last_n_chats(self, num_chats: Optional[int] = None) -> str:
        """Returns the last n chats between the user and assistant.
        Example:
            - To get the last chat, use num_chats=1.
            - To get the last 5 chats, use num_chats=5.
            - To get all chats, use num_chats=None.
            - To get the first chat, use num_chats=None and pick the first message.
        :param num_chats: The number of chats to return.
            Each chat contains 2 messages. One from the user and one from the assistant.
        :return: A list of dictionaries representing the chat history.
        """
        history: List[Dict[str, Any]] = []
        all_chats = self.memory.get_chats()
        if len(all_chats) == 0:
            return ""

        chats_added = 0
        for chat in all_chats[::-1]:
            history.insert(0, chat[1].to_dict())
            history.insert(0, chat[0].to_dict())
            chats_added += 1
            if num_chats is not None and chats_added >= num_chats:
                break
        return json.dumps(history)

    def search_knowledge_base(self, query: str) -> Optional[str]:
        """Search the knowledge base for information about a users query.

        :param query: The query to search for.
        :return: A string containing the response from the knowledge base.
        """
        return self.get_references_from_knowledge_base(query=query)

    ###########################################################################
    # Print Response
    ###########################################################################

    def print_response(self, message: str, stream: bool = True) -> None:
        from phi.cli.console import console
        from rich.live import Live
        from rich.table import Table
        from rich.box import ROUNDED
        from rich.markdown import Markdown

        if stream:
            response = ""
            with Live() as live_log:
                for resp in self.chat(message, stream=True):
                    response += resp

                    table = Table(box=ROUNDED, border_style="blue")
                    table.add_column("Message")
                    table.add_column(message)
                    md_response = Markdown(response)
                    table.add_row("Response", md_response)
                    live_log.update(table)
        else:
            response = self.chat(message, stream=False)  # type: ignore
            md_response = Markdown(response)
            table = Table(box=ROUNDED, border_style="blue")
            table.add_column("Message")
            table.add_column(message)
            table.add_row("Response", md_response)
            console.print(table)
