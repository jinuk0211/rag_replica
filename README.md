# rag_replica
```python

# Licensed under the MIT license.
class MCTS_Searcher:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        discount: float,
        verbose: bool = False,
    ):
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)  # total reward of each node
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)  # total visit count for each node
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = dict()  # children of each node

        #! explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        self.explored_nodes = set()

        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount

        self.verbose = verbose

        global node_cnt
        node_cnt = 0

    def do_rollout(self, root_node: MCTS_Node, rollout_id: int):
        "Make the tree one layer better. (Train for one iteration.)"
        verbose_print("==> Selecting a node...", self.verbose)
        path_1 = self._select(root_node, rollout_id)
        leaf = path_1[-1]
        verbose_print(f"==> Expanding node {leaf.id}...", self.verbose)
        self._expand(leaf, rollout_id)
        verbose_print(f"==> Simulating node {leaf.id}...", self.verbose)
        path_2 = self._simulate(leaf, rollout_id)
        verbose_print(f"==> Backpropagating...", self.verbose)
        self._backpropagate(path_1 + path_2)
        try:
            return path_2[-1]
        except:
            return path_1[-1]

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            # case 1: a node does not have children, then select the node itself
            if node not in self.parent2children.keys():
                return path

            # case 2: a node has children but not all children have been explored, then randomly select an unexplored child
            # unexplored = set(self.parent2children[node]) - self.explored_nodes   # `set` introduces randomness
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                return path

            # case 3: a node has children and all children have been explored, then select one child and go to the next layer
            node = self._uct_select(node, rollout_id)

    def _expand(self, node: MCTS_Node, rollout_id: int):
        "Update the `children` dict with the children of `node`"
        if node in self.explored_nodes:
            return  # already expanded

        if node.is_terminal():
            self.explored_nodes.add(node)
            return  # terminal node is non-expandable

        self.parent2children[node] = node.find_children(rollout_id)

#---------------------------
@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SUBQUESTION = "SUBQUESTION"
    RE_SUBANSWER = "RE_SUBANSWER"
    OST_STEP = "OST_STEP"
    RAG_STEP = "RAG_STEP"
    RE_RAGANSWER = "RE_RAGANSWER"

 def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children
       def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)

            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASED_USER_QUESTION
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        def do_action_generate_subquestions():
            verbose_print(f"---- Generating subquestions for node {self.id}...", self.verbose)

            #! ACTION: generate new subquestions
            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    value = 0.01
                    # breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_re_subanswers():
            verbose_print(f"---- Generating re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers),
                    )
                )
        
        def do_action_generate_rag_and_re_subanswers():
            verbose_print(f"---- Generating rag and re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_rag_and_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_rephrased_user_question():
            verbose_print(f"---- Generating rephrased user question for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rephrased_user_question,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_ost_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_question_retrieve():
            verbose_print(f"---- Generating question retrieve steps for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            retrieved_user_question_list, potential_answers_list = self.generator.generate_user_question_retrieve(
                user_question=self.user_question
            )
            for retrieved_user_question, potential_answers in zip(retrieved_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=retrieved_user_question, 
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_rag_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating rag-step steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_rag_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            print(f"rag step: {ost_step_list}")
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                    do_action_generate_question_retrieve()
                # futures.append(executor.submit(do_action_generate_question_retrieve))
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                if not self.disable_a5:
                    do_action_generate_rephrased_user_question()
                
        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                do_action_generate_direct_answers()
                do_action_generate_subquestions()

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.SUBQUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step(True)
                do_action_generate_re_subanswers()

                # 等待所有任务执行完毕
                if not self.disable_rag:
                    do_action_generate_rag_step(True)

                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                # futures.append(executor.submit(do_action_generate_re_subanswers))
              
        elif self.node_type is Node_Type.RE_SUBANSWER:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step(True)
                if not self.disable_rag:
                    do_action_generate_rag_step(True)
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                
        elif self.node_type is Node_Type.OST_STEP:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_rag:
                    do_action_generate_rag_step()
                if not self.disable_a1:
                    do_action_generate_ost_step()
                do_action_generate_direct_answers()
                

        assert self.children
        return self.children
#--------------------------------------
    def _simulate(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Returns the reward for a random simulation (to completion) of `node`"
        path = []
        cur_node = node
        while True:
            if cur_node.is_terminal():
                self.explored_nodes.add(node)
                return path

            if cur_node not in self.parent2children.keys():
                self.parent2children[cur_node] = cur_node.find_children(rollout_id)

            cur_node = random.choice(self.parent2children[cur_node])  # randomly select a child
            path.append(cur_node)

    def _backpropagate(self, path: List[MCTS_Node]):
        "Send the reward back up to the ancestors of the leaf"
        leaf = path[-1]
        reward = leaf.calculate_reward()
        for node in reversed(path):
            self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "const":
            return self.exploration_weight
#--------------------------ROLLOUT_id 받는지점 - i
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
    def _uct_select(self, node: MCTS_Node, rollout_id: int):
        "Select a child of node, balancing exploration & exploitation"
#---------------------------------------------
        # All children of the node should already be expanded
        assert all(n in self.explored_nodes for n in self.parent2children[node])

        return max(
            self.parent2children[node], key=lambda n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
        )

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
                return 999
            else:
                weight = self._get_weight(rollout_id)
                return self.Q[node] / self.N[node] + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator
        if not args.disable_rag:
            self.retriever = Retriever()
            self.retriever.regist_io_system(self.io)

        self.num_subquestions = args.num_subquestions
        self.num_a1_steps = args.num_a1_steps
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score

        self.mcts_num_last_votes = args.mcts_num_last_votes

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
        self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)

    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }

    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                io_output_list
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def _fewshot_cot_answer_question_with_external_knowledge(self, question: str, external_knowledge: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += "\n\n" + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        io_input += (
               f"Context: {external_knowledge}"
            )
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        return io_input, cleaned_io_output_list

    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
#-----------------------------------------
def concat_subqs_and_subas(solution_trace: Dict[int, Dict[str, str]], question_index: int) -> Tuple[str, int]:
    """Return: concatenated subqs and suba, next subquestion id"""
    solution_trace_str = ""

    for subquestion_id, solution_step in solution_trace.items():
        if subquestion_id == 0:
            continue

        assert subquestion_id > 0
        assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

        solution_trace_str += f"Question {question_index}." + str(subquestion_id) + ": " + solution_step["subquestion"]
        solution_trace_str += "\n"
        solution_trace_str += (
            f"Answer {question_index}." + str(subquestion_id) + ": " + solution_step["subanswer"]["text"]
        )
        solution_trace_str += "\n"

    next_subquestion_id = int(sorted(solution_trace.keys())[-1]) + 1
    return solution_trace_str, next_subquestion_id
#-----------------------------------------
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + "\n"
            + f"The text you generate must start with the string of subquestion index Question {self.question_index}.{next_subquestion_id}:."
        )

        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "\n",
                "The answer",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = list(set([o.strip() for o in io_output_list if o.startswith(f"Question {self.question_index}.{next_subquestion_id}:")]))
        if len(subquestion_list) < 1:
            subquestion_list = list(set([o.strip() for o in io_output_list]))
        print(f"subquestion list: {subquestion_list}")


        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Please use one complete sentence to answer the question: {self.question_index}.{next_subquestion_id}."
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes
#--------------------------------------
def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None
    if "Now we can answer" in subquestion:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True
    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower():
        return True
    return False
def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem


#--------------------------------------
        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=['\n\n\n',
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        print(f"subquestion answer: {subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )
#-------------------------------------
def make_response_prefix(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type, new_subq=None, new_suba=None, new_ost_step=None
) -> str:
    if node_type in [Node_Type.SUBQUESTION, Node_Type.RE_SUBANSWER]:
        response_prefix = ""
        answer_marker = "The answer is"  # todo: hard code "The answer is"
        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            response_prefix += solution_step["subanswer"]["text"].split(answer_marker)[0]
            response_prefix += " "

        if new_subq is not None and new_suba is not None:
            response_prefix += new_suba.split(answer_marker)[0]

        response_prefix = response_prefix.strip(" ")
    elif node_type is Node_Type.OST_STEP:
        response_prefix = ""

        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        if "ost_step" in last_tuple_recording.keys():
            for step_id, step_text in last_tuple_recording["ost_step"].items():
                response_prefix += step_text + " "

        if new_ost_step is not None:
            response_prefix += new_ost_step

        response_prefix = response_prefix.strip(" ")
    elif node_type is None and solution_trace is None:
        response_prefix = ""
    else:
        raise ValueError(f"Invalid node type: {node_type}.")
    think = "Let's think step by step. "
    return think + response_prefix if think not in response_prefix else response_prefix
#---------------------------------------
                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context = user_question

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context} {last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        print(f"re subanswer: {re_subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rag_and_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context = user_question

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context}\n\n{last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )

        print(f"rag subquestion 1: {question}")

        retrieved_context = self.retriever.retrieve(question)

        question = (
            f"{user_question_context} {last_subquestion}\n\n### Relevant Context:\n{retrieved_context}."
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        print(f"rag subquestion 2: {question}")

        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
            most_likely_answer = [f"{answer.strip().strip('\n')}\n\n### Relevant Context: {retrieved_context}\n" for answer in most_likely_answer]
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        print(f"rag subq answer {re_subanswer_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Rephrase Original Question: " + user_question + "\n"
        io_input += "Rephrased question you generate should start with Given a list of conditions, please answer the question. Condition 1:, and it should be one line"
        io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=[])[0]
        io_output = "Given a list of conditions, please answer the question: " + user_question + " Condition 1:" + io_output.split("Condition 1:")[-1] if "Condition 1:" in io_output else "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        print(f"Rephrased user question is: {rephrased_user_question_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer (arabic numerals) is"
                for z in potential_score_output
            ]
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_user_question_retrieve(self, user_question: str):
        rephrased_user_question_list = []

        retrieved_context = self.retriever.retrieve(user_question)

        io_output = f"Given additional informations, please answer the question.\n### Relevant Context: {retrieved_context}\nUser Question: {user_question}." 
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer is"
                for z in potential_score_output
            ]
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_rag_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        
            if next_ost_step_id == 1:
                return self.generate_ost_step(user_question=user_question, solution_trace=solution_trace, paraphrased=paraphrased, parent_is_subquestion=parent_is_subquestion)

        retrieve_question = f"{user_question}\n\n{existing_ost_steps}"
        retrieved_context = self.retriever.retrieve(retrieve_question)

        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples="",
                instruction=user_question,
            )
            + existing_ost_steps
            + "\n"
            + f"### Relevant Context:\n{retrieved_context}\n\n" 
            + f"The text you generate must start with string of current step index Step {next_ost_step_id}:"
        )
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=['\n\n\n', f'Step {next_ost_step_id+1}',str(next_ost_step_id+1)]
        )
        ost_step_list = list(set([f"{io_output.strip().strip('\n')}\n\n### Relevant Context: {retrieved_context}\n" for io_output in io_output_list if io_output.startswith(f"Step {next_ost_step_id}")]))
        if len(ost_step_list) < 1:
            ost_step_list = list(set([f"Step {next_ost_step_id}: {io_output.strip().strip('\n')}" for io_output in io_output_list]))
        print(f"rag step list {ost_step_list}")

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        return ost_step_list, potential_answers_list
    
    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)

        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)        
        
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples='',
                instruction=user_question,
            )
            + existing_ost_steps
            + '\n'
            + f"The text you generate must start with the string Step {next_ost_step_id}:\n"
        )

        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=[f"Step {next_ost_step_id+1}", "\n\n\n"]
        )

        ost_step_list = list(set([io_output.strip().strip('\n') for io_output in io_output_list if io_output.startswith(f"Step {next_ost_step_id}")]))
        if len(ost_step_list)<1:
            ost_step_list = list(set([f"Step {next_ost_step_id}: {io_output.strip().strip('\n')}" for io_output in io_output_list]))

        assert(len(ost_step_list)>0)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=[str(next_ost_step_id+1)],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=[str(next_ost_step_id+1)],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        return ost_step_list, potential_answers_list

    '''def generate_rag_step(self, user_question: str, paraphrased: bool, hint: str):
        #! generate retrieve query
        rag_prompt = (
            "Given the following question, generate a concise and effective search query to retrieve relevant knowledge:\n"
            f"Question {self.question_index}: {user_question}"
            "Query:"
        )

        io_output_list = self.io.generate(
            rag_prompt,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
            ],
        )

        print("rag query is: " + ", ".join(io_output_list))
        #! do retrieve and get answer
        retrieved_documents = self.retriever.retrieve(io_output_list[0])

        direct_answer_list, value_list = [], []

        return retrieved_documents, value_list'''

    def generate_rag_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        #! generate retrieve query
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )

        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        #! do retrieve and get answer
        retrieved_documents = self.retriever.retrieve(io_output_list[0])

        #! merge into question

        #! use retrieved result to reanswer question



        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = [o.strip() for o in io_output_list]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=[
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )
                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix

                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list
    
```

```python
# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm

sys.path.append(".")

from common.arguments import get_parser, post_process_args, save_args
from run_src.mcts_utils import GeneratorError
from run_src.MCTS_for_reasoning_with_rag import Generator, search_for_answers
from Evaluator import *


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def setup_model_parallel() -> Tuple[int, int]:
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size
def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
    
def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model

        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api == "gpt-4o":
        from models.OpenAI_API import load_OpenAI_model 

        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    generator = Generator(args, tokenizer, model, evaluator)

    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < args.start_idx or i >= args.end_idx:
            continue

        problem_id, problem, gt_solution = data_item["id"], data_item["problem"], data_item["solution"]
        if not args.disable_rag:
            evidence = data_item.get("evidence", "")
            generator.retriever.add_evidence(evidence)
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

만약 GSM8Kevaluator
    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.split("#### ")[-1].strip()
#--------------------------
        js = {
            "id": problem_id,
            "problem": problem,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        model_solutions, stopping_id, model_all_solutions = [], -1, []

        # try:
        search_start_time = time.time()
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=i, gt_answer=gt_solution, generator=generator
        )
#---------------------------------------------------------
def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_question=user_question,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        enable_potential_score=args.enable_potential_score,
        disable_rag=args.disable_rag
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)

        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
            root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)
#------------------------
def stochastic_find_best_solution(
    root_node,
    evaluator,
    enable_potential_score,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(evaluator.check_answers_equiv(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None)
    top_answer, top_completion, top_completion_id, top_confidence = evaluator.stochastic_find_most_confident_answer(
        completions=solutions, prior_weights=prior_weights
    )
    return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes, solutions
def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []
    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)
    return valid_solution_nodes
#-----------------------------------------
        if args.save_tree:
            with open(
                os.path.join(
                    args.answer_sheets_dir,
                    f"Question {question_id:04d} - Rollout {i}.tree",
                ),
                "w",
            ) as f:
                print_tree_from_root(
                    mcts_searcher=mcts_searcher,
                    rollout_id=i,
                    root_node=root_node,
                    chosen_node=chosen_node,
                    file=f,
                )

    #! record final traces
    js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
        json.dump(js, f)

    js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
        json.dump(js2, f)

    if args.enable_potential_score:
        js = [node.potential_answers_history for node in all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
            json.dump(js, f)

    print(model_solutions)
    print(model_all_solutions)

    return model_solutions, i, model_all_solutions

class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        disable_a5: bool = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_a1: bool = None,
        # -----------------------------------
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # -------------------------------------------
        # --- For instantiating RAG_STEP node ---
        rag_docs: dict = None,
        # -------------------------------------------
        # --- For instantiating RE_RAGANSWER node ---
        re_raganswer: str = None,

        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
        disable_rag: bool = None
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_a5, user_question, expected_answer, max_depth_allowed, disable_a1]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,generator,disable_a5,user_question,expected_answer,
                        direct_answer,subquestion,subanswer,
                        is_new_subquestion,
                        re_subanswer,ost_step,
                        max_depth_allowed,disable_a1,
                        disable_rag
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_rag
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_rag
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.RE_SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_rag
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, re_subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                        disable_rag
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_a5 = disable_a5
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.disable_a1 = disable_a1
            self.disable_rag = disable_rag
            self.enable_potential_score = enable_potential_score
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_a1 = parent.disable_a1
            self.disable_rag = parent.disable_rag
            self.enable_potential_score = parent.enable_potential_score

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            elif node_type is Node_Type.RE_SUBANSWER:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert self.solution_trace[self.subquestion_counter]["subquestion"] == parent.subquestion
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            elif node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
            Node_Type.RAG_STEP: "RT",
            Node_Type.RE_RAGANSWER: "RR"
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)

            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASED_USER_QUESTION
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        def do_action_generate_subquestions():
            verbose_print(f"---- Generating subquestions for node {self.id}...", self.verbose)

            #! ACTION: generate new subquestions
            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    value = 0.01
                    # breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_re_subanswers():
            verbose_print(f"---- Generating re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers),
                    )
                )
        
        def do_action_generate_rag_and_re_subanswers():
            verbose_print(f"---- Generating rag and re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_rag_and_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_rephrased_user_question():
            verbose_print(f"---- Generating rephrased user question for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rephrased_user_question,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_ost_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_question_retrieve():
            verbose_print(f"---- Generating question retrieve steps for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            retrieved_user_question_list, potential_answers_list = self.generator.generate_user_question_retrieve(
                user_question=self.user_question
            )
            for retrieved_user_question, potential_answers in zip(retrieved_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=retrieved_user_question, 
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_rag_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating rag-step steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_rag_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            print(f"rag step: {ost_step_list}")
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                    do_action_generate_question_retrieve()
                # futures.append(executor.submit(do_action_generate_question_retrieve))
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                if not self.disable_a5:
                    do_action_generate_rephrased_user_question()
                
        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step()
                if not self.disable_rag:
                    do_action_generate_rag_step()
                do_action_generate_direct_answers()
                do_action_generate_subquestions()

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.SUBQUESTION:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step(True)
                do_action_generate_re_subanswers()

                # 等待所有任务执行完毕
                if not self.disable_rag:
                    do_action_generate_rag_step(True)

                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                # futures.append(executor.submit(do_action_generate_re_subanswers))
              
        elif self.node_type is Node_Type.RE_SUBANSWER:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_a1:
                    do_action_generate_ost_step(True)
                if not self.disable_rag:
                    do_action_generate_rag_step(True)
                do_action_generate_direct_answers()
                do_action_generate_subquestions()
                
        elif self.node_type is Node_Type.OST_STEP:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                # 提交所有无依赖任务到线程池
                if not self.disable_rag:
                    do_action_generate_rag_step()
                if not self.disable_a1:
                    do_action_generate_ost_step()
                do_action_generate_direct_answers()
                

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION
#--------------------------------------------------------------
        search_end_time = time.time()
        print(f"Process question {i} cost: {search_end_time - search_start_time}s")
        # except GeneratorError as e:
        #     print(e)
        #     js["generator_error"] = {
        #         "source": e.source,
        #         "io_input": e.io_input,
        #         "io_output_list": e.io_output_list,
        #     }
        # except Exception as e:
        #     print(e)
        #     js["other_error"] = {"text": str(e)}

        num_tested += 1

        with open(os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w") as f:
            json.dump(js, f)

        with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
            f.write(
                f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
            )
            f.write(
                f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
            )

    end_time = time.time()

    print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
    print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
    print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
        f.write(
            f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
        )
        f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")


if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions", type=int, default=2, help="Number of trials for proposing the next subquestion"
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=3)
    parser.add_argument("--disable_a1", action="store_true")
    parser.add_argument("--disable_rag", action="store_true")


    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 32

    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 3

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    if not args.disable_a5:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    save_args(args)
    main(args)
```
