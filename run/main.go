package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/cuberl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/gocube"
	"github.com/unixpickle/serializer"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var netFile string
	var scramble string
	var episodeLen int
	var greedy bool
	var objective cuberl.Objective

	flag.StringVar(&netFile, "net", "../train/out_net", "network file")
	flag.StringVar(&scramble, "scramble", "", "cube scramble")
	flag.IntVar(&episodeLen, "len", 50, "episode length")
	flag.BoolVar(&greedy, "greedy", false, "choose actions greedily")
	flag.Var(&objective, "objective", cuberl.ObjectiveUsage)

	flag.Parse()

	if scramble == "" {
		essentials.Die("Missing -scramble flag. See -help for more.")
	}

	var net anyrnn.Block
	if err := serializer.LoadAny(netFile, &net); err != nil {
		essentials.Die("read network:", err)
	}

	parsed, err := gocube.ParseMoves(scramble)
	if err != nil {
		essentials.Die("parse scramble:", err)
	}

	cube := gocube.SolvedCubieCube()
	for _, m := range parsed {
		cube.Move(m)
	}

	state := &cuberl.State{Cube: cube, Objective: objective}
	state.MaxSolved = state.NumSolved()
	results := cuberl.PolicyMoves(net, state, episodeLen, greedy)

	for _, m := range results {
		var rew float64
		state, rew = state.Move(m)
		fmt.Print(m.String() + " ")
		for i := 0; i < int(rew); i++ {
			fmt.Print("|")
		}
		if rew > 0 {
			fmt.Print(" ")
		}
	}
	fmt.Println()
}
