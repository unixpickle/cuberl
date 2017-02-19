package main

import (
	"flag"
	"log"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/cuberl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	var stepSize float64
	var batchSize int
	var netFile string
	var hiddenSize int
	var fetcher Fetcher

	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.Float64Var(&fetcher.Baseline, "baseline", 2.33, "policy gradient reward baseline")
	flag.IntVar(&batchSize, "batch", 10, "SGD batch size")
	flag.IntVar(&hiddenSize, "hidden", 128, "LSTM state size for new agents")
	flag.StringVar(&netFile, "net", "out_net", "network file path")
	flag.IntVar(&fetcher.EpisodeLen, "len", 50, "episode length")

	flag.Parse()

	if err := serializer.LoadAny(netFile, &fetcher.Agent); err != nil {
		log.Println("Creating new network...")
		fetcher.Agent = cuberl.NewAgent(anyvec32.CurrentCreator(), hiddenSize)
	} else {
		log.Println("Loaded network.")
	}

	log.Println("Training...")
	tr := &anys2s.Trainer{
		Func: func(sq anyseq.Seq) anyseq.Seq {
			return anyrnn.Map(sq, fetcher.Agent)
		},
		Cost:    anynet.DotCost{},
		Params:  fetcher.Agent.(anynet.Parameterizer).Parameters(),
		Average: true,
	}
	var iter int
	sgd := &anysgd.SGD{
		Samples:     anysgd.LengthSampleList(batchSize),
		Gradienter:  tr,
		Fetcher:     &fetcher,
		Transformer: &anysgd.Adam{},
		BatchSize:   batchSize,
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			qual := sampleQuality(fetcher.Agent, fetcher.EpisodeLen)
			log.Printf("iter %d: cost=%v nsolved=%d", iter, tr.LastCost, qual)
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())

	log.Println("Saving...")
	if err := serializer.SaveAny(netFile, fetcher.Agent); err != nil {
		essentials.Die(err)
	}
}

type Fetcher struct {
	Agent      anyrnn.Block
	Baseline   float64
	EpisodeLen int
}

func (f *Fetcher) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	return cuberl.Samples(f.Agent, cuberl.RandomStates(s.Len()), f.EpisodeLen,
		f.Baseline), nil
}

func sampleQuality(agent anyrnn.Block, episodeLen int) int {
	state := cuberl.RandomStates(1)[0]
	moves := cuberl.AgentMoves(agent, state, episodeLen, false)
	for _, x := range moves {
		state, _ = state.Move(x)
	}
	return state.MaxSolved
}
