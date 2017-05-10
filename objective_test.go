package cuberl

import (
	"testing"

	"github.com/unixpickle/gocube"
)

func TestObjective(t *testing.T) {
	moves := []string{
		"R U R' U' F' U' F",
		"U2 L2",
		"U2 L2 R2",
		"U2 L2 R2 F2",

		"R2 U2",
		"R2 U2 L2",
		"R2 U2 L2 U2 L2",
		"R2 U2 L2 F2",
		"R2 U2 L2 U2 R2 D2",

		"L2 B' U' F2 D2 L' R",
		"L2 B' U' F2 D2 L' R F2",
		"L2 B' U' F2 D2 L' R F2 L' U",
	}
	scores := []int{
		8, 5, 2, 1,
		7, 2, 4, 1, 3,
		0, 1, 3,
	}
	objs := []Objective{
		FirstLayer, FirstLayer, FirstLayer, FirstLayer,
		PetrusBlock, PetrusBlock, PetrusBlock, PetrusBlock, PetrusBlock,
		FullCube, FullCube, FullCube,
	}
	for i, scramble := range moves {
		cube := gocube.SolvedCubieCube()
		moves, err := gocube.ParseMoves(scramble)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range moves {
			cube.Move(m)
		}
		expected := scores[i]
		actual := objs[i].Evaluate(&cube)
		if actual != expected {
			t.Errorf("scramble %d: got %d expected %d", i, actual, expected)
		}
	}
}
