package main

import (
	"flag"
	"log"
	"math"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/cuberl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/anyplugin"
)

func main() {
	var stepSize float64
	var batchSize int
	var netFile string
	var hiddenSize int
	var epLen int
	var objective cuberl.Objective
	var cgIters int
	var layers int
	var lowmem bool
	var nprnn bool

	flag.Float64Var(&stepSize, "step", 0.01, "TRPO step size")
	flag.IntVar(&cgIters, "cgiters", 10, "CG iterations for TRPO")
	flag.IntVar(&batchSize, "batch", 10, "experience batch size")
	flag.IntVar(&hiddenSize, "hidden", 128, "LSTM state size for new agents")
	flag.IntVar(&layers, "layers", 1, "number of LSTM layers")
	flag.StringVar(&netFile, "net", "out_net", "network file path")
	flag.IntVar(&epLen, "len", 20, "episode length")
	flag.Var(&objective, "objective", cuberl.ObjectiveUsage)
	flag.BoolVar(&lowmem, "lowmem", false, "use a memory-saving algorithm")
	flag.BoolVar(&nprnn, "nprnn", false, "use npRNN instead of LSTM")

	flag.Parse()

	creator := anyvec32.CurrentCreator()

	var policy anyrnn.Block
	if err := serializer.LoadAny(netFile, &policy); err != nil {
		log.Println("Creating new network...")
		policy = cuberl.NewPolicy(creator, hiddenSize, layers, nprnn)
	} else {
		log.Println("Loaded network.")
	}

	envs := make([]anyrl.Env, batchSize)
	for i := range envs {
		envs[i] = &cuberl.Env{
			Objective: objective,
			EpLen:     epLen,
		}
	}

	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      anynet.AllParameters(policy),
			ActionSpace: anyrl.Softmax{},
			Iters:       cgIters,

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				if lowmem {
					interval := essentials.MaxInt(1, int(math.Sqrt(float64(epLen))))
					out := lazyrnn.FixedHSM(interval, true, seq, b)
					return lazyseq.Lazify(lazyseq.Unlazify(out))
				} else {
					return lazyseq.Lazify(anyrnn.Map(lazyseq.Unlazify(seq), b))
				}
			},
		},
		TargetKL: stepSize,
	}
	roller := &anyrl.RNNRoller{Block: policy, ActionSpace: anyrl.Softmax{}}

	r := rip.NewRIP()
	var batchIdx int
	for !r.Done() {
		batch, err := roller.Rollout(envs...)
		if err != nil {
			panic(err)
		}
		log.Printf("batch %d: reward=%f", batchIdx, batch.Rewards.Mean())
		batchIdx++
		if r.Done() {
			break
		}
		trpo.Run(batch).AddToVars()
	}

	log.Println("Saving...")
	if err := serializer.SaveAny(netFile, policy); err != nil {
		essentials.Die(err)
	}
}
