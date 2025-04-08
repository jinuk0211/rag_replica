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
